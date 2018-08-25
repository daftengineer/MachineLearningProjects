import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics


val df2 = spark.read.format("csv").option("header","true").option("inferSchema","true").load("hdfs://nn01.itversity.com:8020/user/pratiksheth/frauddetectiondatafull.csv").cache()
val df = df2.withColumn("isFraud", col("isFraud").cast("double"))
val clean = df.select("isFraud","type", "amount", "oldbalanceOrg", "newbalanceOrig","oldbalanceDest","newbalanceDest").cache()
val typeIndex = new StringIndexer().setInputCol("type").setOutputCol("typeindex")
var strModel = typeIndex.fit(clean)
val cleanandtransformed = strModel.transform(clean).select("isFraud","typeindex", "amount", "oldbalanceOrg", "newbalanceOrig","oldbalanceDest","newbalanceDest").cache()
val vectas = new VectorAssembler().setInputCols(Array("typeindex", "amount", "oldbalanceOrg", "newbalanceOrig","oldbalanceDest","newbalanceDest")).setOutputCol("tmpvect")
val cleanandvectorized = vectas.transform(cleanandtransformed.na.drop)
val normalized = new Normalizer().setInputCol("tmpvect").setOutputCol("features")
val normalizedOutput = normalized.transform(cleanandvectorized.na.drop)
val split = normalizedOutput.randomSplit(Array(0.8,0.2) ,seed = 13L)
val training = split(0)
val test = split(1)
training.describe()
val lr = new LogisticRegression().setMaxIter(10)
lr.setLabelCol("isFraud")
val model = lr.fit(training)
var result = model.transform(test)
result.describe()
result = result.select("prediction","isFraud")
val predictionAndLabels = result.map { row =>
       (row.get(0).asInstanceOf[Double],row.get(1).asInstanceOf[Double])
     } 
val prednlabrdd = predictionAndLabels.rdd
val metrics = new BinaryClassificationMetrics(prednlabrdd)
println("Accuracy is: " + metrics.areaUnderROC())


