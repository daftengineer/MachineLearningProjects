import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.linalg.{ Vector,Vectors }
import org.apache.spark.sql.functions.col


object FraudDetectionWithGBT{

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("Fraud Detection With GBT")
      .master("yarn")
      .getOrCreate()
    import spark.implicits._
    val df2 =spark.read.format("csv").option("header","true").option("inferSchema","true").
      load("hdfs://nn01.itversity.com:8020/user/pratiksheth/frauddetectiondata.csv")
    val df = df2.withColumn("isFraud", col("isFraud").cast("double"))
    val clean = df.select("isFraud","type", "amount", "oldbalanceOrg", "newbalanceOrig","oldbalanceDest","newbalanceDest").cache()
    val typeIndex = new StringIndexer().setInputCol("type").setOutputCol("typeindex")
    val strModel = typeIndex.fit(clean)
    val cleanandtransformed = strModel.transform(clean)
      .select("isFraud","typeindex", "amount", "oldbalanceOrg", "newbalanceOrig","oldbalanceDest","newbalanceDest").cache()
    val newVectRdd = cleanandtransformed
      .map(row =>
        (row.get(0).asInstanceOf[Double],Vectors.dense(row.get(1).asInstanceOf[Double],row.get(2).asInstanceOf[Double],row.get(3).asInstanceOf[Double],row.get(4).asInstanceOf[Double],row.get(5).asInstanceOf[Double],row.get(6).asInstanceOf[Double])))
    val splits = newVectRdd.randomSplit(Array(0.75,0.25) ,seed = 12L)
    val (trainingData, testData) = (splits(0), splits(1))
    val trainingDataLabelPoint = trainingData.map(row => LabeledPoint(row._1.asInstanceOf[Double],row._2.asInstanceOf[Vector])).rdd
    val booststrat = BoostingStrategy.defaultParams("Classification")
    booststrat.numIterations = 12
    booststrat.treeStrategy.numClasses = 2
    booststrat.treeStrategy.maxDepth = 15
    booststrat.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    val model = GradientBoostedTrees.train(trainingDataLabelPoint,booststrat)
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point._2)
      (point._1, prediction)
    }
    val MSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.rdd.mean()
    println(s"Test Mean Squared Error = $MSE")
    model.save(spark.sparkContext,"hdfs://nn01.itversity.com:8020/user/pratiksheth/tmp/myGradientBoostingRegressionModel")

  }
}
