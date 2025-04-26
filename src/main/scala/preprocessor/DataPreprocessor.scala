package spark.kp4.preprocessor

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

class DataPreprocessor(spark: SparkSession) {
  val featureColumns = Seq(
    "TransactionAmt", "EVENT_TIMESTAMP", "ENTITY_ID", "ProductCD", "DeviceType", "DeviceInfo",  "addr1", "dist1",
    "card1", "card2", "card3", "card5", "card6",
    "C1", "C2", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14", "V62", "V70", "V76", "V78", "V82", "V91", "V127", "V130",
    "V139", "V160", "V165", "V187", "V203", "V207", "V209", "V210", "V221", "V234", "V257", "V258",
    "V261", "V264", "V266", "V267", "V271", "V274", "V277", "V283", "V285", "V289", "V291", "V294",
    "id_01", "id_02", "id_05", "id_06", "id_09", "id_13", "id_17", "id_19", "id_20"
  )

  def isDataProcessed(): Boolean = {
    new java.io.File("data/processed/train/processed_train_data.csv").exists
  }

  def preProcessData(dataType: String): Unit = {
    val transactions = spark.read.option("header", "true").option("inferSchema", "true").csv(s"data/${dataType}_transaction.csv")
    val identities = spark.read.option("header", "true").option("inferSchema", "true").csv(s"data/${dataType}_identity.csv")

    val fullData = transactions.join(identities, Seq("TransactionID"), "left_outer")
    val processedData = normalizeDColumns(fullData)
      .transform(fillNa)
      .transform(createEventTimestamp)
      .transform(createEntityId)
      .transform(selectFeatures)

    saveProcessedData(processedData, dataType)
  }


  def selectFeatures(df: DataFrame): DataFrame = {
    val updatedFeatureColumns = if (df.columns.contains("isFraud")) {
      featureColumns :+ "isFraud"
    } else {
      featureColumns
    }
    df.select(updatedFeatureColumns.map(col): _*)
  }

  private def normalizeDColumns(df: DataFrame): DataFrame = {
    (1 to 15).foldLeft(df) { (tempDf, i) =>
      if (Seq(1, 2, 3, 5, 9).contains(i)) tempDf
      else tempDf.withColumn(s"D$i", col(s"D$i") - col("TransactionDT") / lit(24 * 60 * 60))
    }
  }

  private def fillNa(df: DataFrame): DataFrame = {
    df.na.fill("unknown").na.fill(0)
  }

  def createEntityId(df: DataFrame): DataFrame = {
    val withCardAddr = df.withColumn("card1_addr1", concat_ws("_", col("card1"), col("addr1")))
    val withDay = withCardAddr.withColumn("day", col("TransactionDT") / lit(24 * 60 * 60))
    withDay.withColumn("ENTITY_ID",
      concat_ws("_",
        col("card1_addr1"),
        floor(col("day") - col("D1")).cast("string")
      ))
  }

  def createEventTimestamp(df: DataFrame): DataFrame = {
    df.withColumn("EVENT_TIMESTAMP", lit(1609498925) + col("TransactionDT"))
  }

  private def saveProcessedData(df: DataFrame, dataType: String): Unit = {
    val tempPath = s"data/processed/${dataType}/temp_processed_${dataType}_data"
    val finalPath = s"data/processed/${dataType}/processed_${dataType}_data.csv"

    df
      .coalesce(1)
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv(tempPath)

    // Отримуємо доступ до файлової системи
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)

    // Знаходимо файл part-00000
    val partFile = fs.globStatus(new Path(s"$tempPath/part-*"))(0).getPath

    // Перейменовуємо файл
    fs.rename(partFile, new Path(finalPath))

    // Видаляємо тимчасову папку
    fs.delete(new Path(tempPath), true)
  }
}
