package spark.kp4

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

import java.io.File


object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("AntiFraud")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .master("local[*]")
      .getOrCreate()

    //if data/processed_train_data.csv exists, run preProcessData if not do nothing
    if (!new File("data/processed_train_data.csv").exists) {
      preProcessData(spark)
    }

    trainModel(spark)
  }

  private def trainModel(spark: SparkSession): Unit = {
    val data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/processed_train_data.csv")

    // 1. Перетворення даних
    // Перетворення мітки (isFraud) у числовий формат
    val labelIndexer = new StringIndexer()
      .setInputCol("EVENT_LABEL")
      .setOutputCol("indexedLabel")
      .fit(data)

    // Збір усіх фіч у вектор
    val featureColumns = data.columns.filterNot(col =>
      Seq("EVENT_LABEL", "EVENT_TIMESTAMP", "ENTITY_ID", "LABEL_TIMESTAMP").contains(col)
    )

    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val assembledData = assembler.transform(labelIndexer.transform(data))

    // 3. Ініціалізація та тренування Random Forest
    val randomForest = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("features")
      .setNumTrees(100) // Кількість дерев

    val model = randomForest.fit(assembledData)

    // 4. Передбачення на тестових даних
    val predictions = model.transform(testData)

    // 5. Оцінка моделі
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(s"Точність моделі: $accuracy")

    // Збереження моделі
    model.write.overwrite().save("models/random_forest_model")

    spark.stop()
  }

  private def preProcessData(spark: SparkSession): Unit = {
    // Читання даних
    val trainTransactions = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/train_transaction.csv")

    val trainIdentities = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/train_identity.csv")

    // Об'єднання таблиць за TransactionID
    val fullData = trainTransactions.join(trainIdentities, Seq("TransactionID"), "left_outer")

    // 1. Нормалізація D колонок
    val normalizedData = normalizeDColumns(fullData)

    // 2. Створення ENTITY_ID
    val withEntityId = createEntityId(normalizedData)

    // 3. Обробка TransactionDT у EVENT_TIMESTAMP
    val withTimestamps = createEventTimestamp(withEntityId)

    // 4. Вибір релевантних фіч
    val selectedFeatures = selectFeatures(withTimestamps)

    // 5. Збереження результатів
    selectedFeatures
      .coalesce(1)
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv("data/temp_processed_train_data")

    val tempPath = "data/temp_processed_train_data"
    val finalPath = "data/processed_train_data.csv"

    // Отримуємо доступ до файлової системи
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)

    // Знаходимо файл part-00000
    val partFile = fs.globStatus(new Path(s"$tempPath/part-*"))(0).getPath

    // Перейменовуємо файл
    fs.rename(partFile, new Path(finalPath))

    // Видаляємо тимчасову папку
    fs.delete(new Path(tempPath), true)
  }

  def normalizeDColumns(df: DataFrame): DataFrame = {
    (1 to 15).foldLeft(df) { (tempDf, i) =>
      if (Seq(1, 2, 3, 5, 9).contains(i)) tempDf
      else tempDf.withColumn(s"D$i", col(s"D$i") - col("TransactionDT") / lit(24 * 60 * 60))
    }
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
    val baseDate = "2021-01-01T00:00:00Z"
    val baseTimestamp = java.sql.Timestamp.valueOf(baseDate.replace("T", " ").replace("Z", ""))

    // Додавання секунд до базової дати через функцію unix_timestamp
    val withEventTimestamp = df.withColumn(
      "EVENT_TIMESTAMP",
      from_unixtime(unix_timestamp(lit(baseDate)) + col("TransactionDT").cast("long"))
    )

    // Додавання LABEL_TIMESTAMP
    withEventTimestamp
      .drop("TransactionDT") // Видалення оригінальної колонки
      .withColumn("LABEL_TIMESTAMP", lit(java.time.LocalDateTime.now.toString))
  }


  def selectFeatures(df: DataFrame): DataFrame = {
    val featureColumns = Seq(
      "TransactionAmt", "ProductCD", "card1", "card2", "card3", "card5", "card6", "addr1", "dist1",
      "P_emaildomain", "R_emaildomain", "C1", "C2", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
      "C11", "C12", "C13", "C14", "V62", "V70", "V76", "V78", "V82", "V91", "V127", "V130",
      "V139", "V160", "V165", "V187", "V203", "V207", "V209", "V210", "V221", "V234", "V257",
      "V258", "V261", "V264", "V266", "V267", "V271", "V274", "V277", "V283", "V285", "V289",
      "V291", "V294", "id_01", "id_02", "id_05", "id_06", "id_09", "id_13", "id_17", "id_19",
      "id_20", "DeviceType", "DeviceInfo", "EVENT_TIMESTAMP", "ENTITY_ID", "LABEL_TIMESTAMP"
    )
    df.select(featureColumns.map(col): _*)
  }
}