from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("hello-world").getOrCreate()

df = spark.createDataFrame([("Luiz",)], ["name"])

df.show()
