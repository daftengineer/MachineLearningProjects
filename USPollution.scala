import org.apache.spark.sql.expressions.MutableAggregationBuffer
import org.apache.spark.sql.expressions.UserDefinedAggregateFunction
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
  class euclid extends UserDefinedAggregateFunction {
    override def inputSchema: org.apache.spark.sql.types.StructType =
      StructType(StructField("value", DoubleType) :: Nil)
    override def bufferSchema: StructType = StructType(
      StructField("buffered", DoubleType) :: Nil
    )
    override def dataType: DataType = DoubleType
    override def deterministic: Boolean = true
    override def initialize(buffer: MutableAggregationBuffer): Unit = {
      buffer(0) = 0.0
    }
    override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
     buffer(0) = buffer.getAs[Double](0) + math.pow(input.getAs[Double](0) , 2)
    }
    override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
      buffer1(0) = buffer1.getAs[Double](0) + buffer2.getAs[Double](0)
    }
    override def evaluate(buffer: Row): Any = {
      math.sqrt(buffer.getDouble(0))
    }
  }
spark.udf.register("edf", new euclid)
val edf = new euclid
val df = spark.read.format("csv").option("header","true").option("inferSchema","true").load("hdfs://nn01.itversity.com/user/pratiksheth/epa_hap_daily_summary.csv")
//1
val dfimp = df.select("date_local","state_name","county_code","parameter_name","units_of_measure","arithmetic_mean","method_name").withColumn("year",year($"date_local")).groupBy("state_name","year","parameter_name").mean("arithmetic_mean").sort("state_name","year","parameter_name")
val f = dfimp.groupBy("state_name","year").agg(edf($"avg(arithmetic_mean)")).sort(asc("year"),desc("euclid(avg(arithmetic_mean))"))
val mostpollutedlist = f.groupBy("year").agg(max("euclid(avg(arithmetic_mean))") as("concentration")).sort("year")
val joined = mostpollutedlist.join(f,mostpollutedlist("concentration") === f("euclid(avg(arithmetic_mean))") && mostpollutedlist("year") === f("year")).select(mostpollutedlist("year") as("per year"),f("state_name"))
joined.sort("per year").show(30,false)

//2
val dfimp = df.select("date_local","city_name","county_code","parameter_name","units_of_measure","arithmetic_mean","method_name").withColumn("year",year($"date_local")).groupBy("city_name","year","parameter_name").mean("arithmetic_mean").sort("city_name","year","parameter_name")
val f = dfimp.groupBy("city_name","year").agg(edf($"avg(arithmetic_mean)")).sort(asc("year"),desc("euclid(avg(arithmetic_mean))"))
val mostpollutedlist = f.groupBy("year").agg(max("euclid(avg(arithmetic_mean))") as("concentration")).sort("year")
val joined = mostpollutedlist.join(f,mostpollutedlist("concentration") === f("euclid(avg(arithmetic_mean))") && mostpollutedlist("year") === f("year")).select(mostpollutedlist("year") as("per year"),f("city_name"))
joined.sort("per year").show(30,false)

//3
val new1 = df.select("parameter_name","state_code","units_of_measure","arithmetic_mean","date_local").withColumn("year",year($"date_local")).withColumn("molar", when(df("parameter_name")==="cis-13-Dichloropropene",111).when(df("parameter_name")==="Benzene",78).when(df("parameter_name")==="Tetrachloroethylene",165).when(df("parameter_name")==="Ethylene dichloride",99).when(df("parameter_name")==="Chloroform",119).when(df("parameter_name")==="12-Dichloropropane",113).when(df("parameter_name")==="Trichloroethylene",131).when(df("parameter_name")==="Acetaldehyde",44).when(df("parameter_name")==="1122-Tetrachloroethane",168).when(df("parameter_name")==="Vinyl chloride",62).when(df("parameter_name")==="Acrolein - Verified",56).when(df("parameter_name")==="Carbon tetrachloride",154).when(df("parameter_name")==="Dichloromethane",85).when(df("parameter_name")==="Acrolein - Unverified",56).when(df("parameter_name")==="Ethylene dibromide",99).when(df("parameter_name")==="13-Butadiene",54).when(df("parameter_name")==="trans-13-Dichloropropene",113).otherwise(0))
val unitname = new1.withColumn("units_of_measure",when(df("units_of_measure")==="Micrograms/cubic meter (25 C)","mg/m3").when(df("units_of_measure")==="Micrograms/cubic meter (LC)","mg/m3").when(df("units_of_measure")==="Parts per billion Carbon","ppb").when(df("units_of_measure")==="Nanograms/cubic meter (25 C)","ng/m3"))
val realconcentration = unitname.withColumn("Concentration", when(unitname("molar") === 0, unitname("arithmetic_mean")).when(unitname("units_of_measure") === "ng/m3",$"arithmetic_mean" / lit(1000)).otherwise(lit(0.0409)*$"arithmetic_mean"*$"molar"))
val maximum = realconcentration.groupBy($"year").max("Concentration")
val joined = maximum.join(realconcentration, realconcentration("Concentration") === maximum("max(Concentration)") && realconcentration("year")=== maximum("year"))
joined.show(30)


//4
val newdata = df.select("date_local","method_name").withColumn("date_local",year($"date_local"))
val counter = newdata.groupBy("date_local","method_name").agg(count("method_name"))
val maxcount = counter.groupBy("date_local").agg(max("count(method_name)"))
val joined = maxcount.join(counter, counter("count(method_name)") === maxcount("max(count(method_name))") && counter("date_local") === maxcount("date_local"))
joined.show(30,false)


