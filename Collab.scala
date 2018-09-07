import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS

object Collab {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("Collaborative")
      .master("yarn")
      .getOrCreate()
    import spark.implicits._
    val df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("hdfs://nn01.itversity.com/user/pratiksheth/smallprocessed.csv")
    val df1 = df.na.drop()
    val Array(trainingdata,testingdata) = df1.randomSplit(Array(0.85,0.15))
    val iter = Array(5,10,15)
    val reg = Array(0.1,0.05,0.01)
    var (i,j)=(0.0,0.0)
    for(i <- iter) {
      for(j <- reg) {
        val als = new ALS().setMaxIter(i).setRegParam(j).setUserCol("userid").setItemCol("movieid").setRatingCol("rating")
        val model = als.fit(trainingdata)
        model.setColdStartStrategy("drop")
        val predictions = model.transform(testingdata)
        val evaluator = new RegressionEvaluator()
          .setMetricName("rmse")
          .setLabelCol("rating")
          .setPredictionCol("prediction")
        val rmse = evaluator.evaluate(predictions)
        println(s"Root-mean-square error = $rmse for iteration $i regparam $j")
      }
    }
  }

}
