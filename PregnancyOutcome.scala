import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.linalg.{ Vector,Vectors }
import org.apache.spark.sql.functions.col

object PregnancyOutcome{

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("Pregnancy")
      .master("yarn")
      .getOrCreate()
    import spark.implicits._
    val inputData =spark.read.format("csv").option("header","true").option("inferSchema","true").option("delimiter","|").
      load("hdfs://nn01.itversity.com:8020/user/pratiksheth/AHS_Woman_08_Rajasthan_small.csv")
    val doubledColumn = inputData.withColumn("outcome_pregnancy", col("outcome_pregnancy").cast("double"))
      .withColumn("district", col("district").cast("double"))
      .withColumn("rural",col("rural").cast("double"))
      .withColumn("age",col("age").cast("double"))
      .withColumn("marital_status",col("marital_status").cast("double"))
      .withColumn("delivered_any_baby",col("delivered_any_baby").cast("double"))
      .withColumn("born_alive_female",col("born_alive_female").cast("double"))
      .withColumn("born_alive_male",col("born_alive_male").cast("double"))
      .withColumn("surviving_female",col("surviving_female").cast("double"))
      .withColumn("surviving_male",col("surviving_male").cast("double"))
      .withColumn("mother_age_when_baby_was_born",col("mother_age_when_baby_was_born").cast("double"))
      .withColumn("is_tubectomy",col("is_tubectomy").cast("double"))
      .withColumn("is_vasectomy",col("is_vasectomy").cast("double"))
      .withColumn("is_copper_t",col("is_copper_t").cast("double"))
      .withColumn("is_pills_daily",col("is_pills_daily").cast("double"))
      .withColumn("is_piils_weekly",col("is_piils_weekly").cast("double"))
      .withColumn("is_emergency_contraceptive",col("is_emergency_contraceptive").cast("double"))
      .withColumn("is_condom",col("is_condom").cast("double"))
      .withColumn("is_contraceptive",col("is_contraceptive").cast("double"))
      .withColumn("is_periodic_abstinence",col("is_periodic_abstinence").cast("double"))
      .withColumn("is_withdrawal",col("is_withdrawal").cast("double"))
      .withColumn("is_amenorrahoea",col("is_amenorrahoea").cast("double"))
      .withColumn("is_currently_pregnant",col("is_currently_pregnant").cast("double"))
      .withColumn("pregnant_month",col("pregnant_month").cast("double"))
      .withColumn("willing_to_get_pregnant",col("willing_to_get_pregnant").cast("double"))
      .withColumn("is_currently_menstruating",col("is_currently_menstruating").cast("double"))
      .withColumn("when_you_bcome_mother_last_time",col("when_you_bcome_mother_last_time").cast("double"))
      .withColumn("aware_abt_hiv",col("aware_abt_hiv").cast("double"))
      .withColumn("aware_of_the_danger_signs",col("aware_of_the_danger_signs").cast("double"))
      .withColumn("religion",col("religion").cast("double"))
      .withColumn("diagnosed_for",col("diagnosed_for").cast("double"))
      .withColumn("smoke",col("smoke").cast("double"))
      .withColumn("chew",col("chew").cast("double"))
      .withColumn("alcohol",col("alcohol").cast("double"))
      .withColumn("water_filteration",col("water_filteration").cast("double"))
      .withColumn("is_husband_living_with_you",col("is_husband_living_with_you").cast("double"))
      .withColumn("compensation_after_ster",col("compensation_after_ster").cast("double"))
    val clean = doubledColumn.select("outcome_pregnancy","district","rural","age","marital_status","delivered_any_baby","born_alive_female","born_alive_male","surviving_female","surviving_male","mother_age_when_baby_was_born","is_tubectomy","is_vasectomy","is_copper_t","is_pills_daily","is_piils_weekly","is_emergency_contraceptive","is_condom","is_contraceptive","is_periodic_abstinence","is_withdrawal","is_amenorrahoea","is_currently_pregnant","pregnant_month","willing_to_get_pregnant","is_currently_menstruating","when_you_bcome_mother_last_time","aware_abt_hiv","aware_of_the_danger_signs","religion","diagnosed_for","smoke","chew","alcohol","water_filteration","is_husband_living_with_you","compensation_after_ster")
    val superclean =clean.na.fill(0)
    val labelPointDataFrameUnclean = superclean
      .map(row =>
        (row.get(0).asInstanceOf[Double],Vectors.dense(row.get(1).asInstanceOf[Double],row.get(2).asInstanceOf[Double],row.get(3).asInstanceOf[Double],row.get(4).asInstanceOf[Double],row.get(5).asInstanceOf[Double],row.get(6).asInstanceOf[Double],row.get(7).asInstanceOf[Double],row.get(8).asInstanceOf[Double],row.get(9).asInstanceOf[Double],row.get(10).asInstanceOf[Double],row.get(11).asInstanceOf[Double],row.get(12).asInstanceOf[Double],row.get(13).asInstanceOf[Double],row.get(14).asInstanceOf[Double],row.get(15).asInstanceOf[Double],row.get(16).asInstanceOf[Double],row.get(17).asInstanceOf[Double],row.get(18).asInstanceOf[Double],row.get(19).asInstanceOf[Double],row.get(20).asInstanceOf[Double],row.get(21).asInstanceOf[Double],row.get(22).asInstanceOf[Double],row.get(23).asInstanceOf[Double],row.get(24).asInstanceOf[Double],row.get(25).asInstanceOf[Double],row.get(26).asInstanceOf[Double],row.get(27).asInstanceOf[Double],row.get(28).asInstanceOf[Double],row.get(29).asInstanceOf[Double],row.get(30).asInstanceOf[Double],row.get(31).asInstanceOf[Double],row.get(32).asInstanceOf[Double],row.get(33).asInstanceOf[Double],row.get(34).asInstanceOf[Double],row.get(35).asInstanceOf[Double],row.get(36).asInstanceOf[Double])))
    val labelPointDataFrame = labelPointDataFrameUnclean.filter("_1 != 0")
    val splits = labelPointDataFrame.randomSplit(Array(0.75,0.25) ,seed = 12L)
    val (trainingData, testData) = (splits(0), splits(1))
    val trainingDataLabelPoint = trainingData.map(row => LabeledPoint(row._1.asInstanceOf[Double],row._2.asInstanceOf[Vector])).rdd
    val iteration = Array(10,12,15,18,20)
    val depth = Array(5,10,15,20,25)
    val boostingPolicy = BoostingStrategy.defaultParams("Classification")
    boostingPolicy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    boostingPolicy.treeStrategy.numClasses = 2
    var (i,j) = (0,0)
    for (i <- iteration) {
      for (j <- depth) {
        boostingPolicy.numIterations = i
        boostingPolicy.treeStrategy.maxDepth = j
        var model = GradientBoostedTrees.train(trainingDataLabelPoint, boostingPolicy)
        var realWithPredicted = testData.map { point =>
          var prediction = model.predict(point._2)
          (point._1, prediction)
        }
        var MSE = realWithPredicted.map { case (real, predicted) => math.pow((real - predicted), 2) }.rdd.mean()
        println(s"Test Mean Squared Error = $MSE for Iteration = $i and depth = $j")
        model.save(spark.sparkContext, "hdfs://nn01.itversity.com:8020/user/pratiksheth/tmp/PregnancyClassificationModel"+s"${i}"+s"${j}")
      }
    }

  }
}