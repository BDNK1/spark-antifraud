package spark.kp4
package preprocessor

import preprocessor.FeaturePreprocessor.schema

import org.apache.spark.ml.feature._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, split}
import org.apache.spark.sql.types._

import java.nio.file.{Files, Paths}

class FeaturePreprocessor {

  val word2VecModel: Word2VecModel = loadWord2VecModel(FeaturePreprocessor.word2VecPath)
  val hashingEntityIdModel: HashingTF = loadHashing(FeaturePreprocessor.hashingEntityPath)
  val indexers: Map[String, StringIndexerModel] = loadAllIndexers(FeaturePreprocessor.indexersPath)
  val assembler: VectorAssembler = loadAssembler(FeaturePreprocessor.assemblerPath)

  def assembleStreamingData(trainData: DataFrame): DataFrame = {
    applyWord2Vec(trainData)
      .transform(applyHashingEntityId)
      .transform(applyStringIndexing)
      .transform(assembleFeatures)
  }

  private def applyWord2Vec(df: DataFrame): DataFrame = {
    val tokenizedData = df.withColumn("DeviceInfoTokens", split(col("DeviceInfo"), " "))
    word2VecModel.transform(tokenizedData)
  }

  private def applyHashingEntityId(df: DataFrame): DataFrame = {
    val tokenizedData = df.withColumn("ENTITY_ID_tokens", split(col("ENTITY_ID"), "_"))

    hashingEntityIdModel.transform(tokenizedData)
  }

  private def applyStringIndexing(df: DataFrame): DataFrame = {
    schema.fields
      .filter(f => f.dataType == StringType && f.name != "DeviceInfo"&& f.name != "ENTITY_ID")
      .map(_.name)
      .foldLeft(df) { (data, colName) =>
        indexers(colName).transform(data)
      }
  }

  private def assembleFeatures(df: DataFrame): DataFrame = {
    assembler.transform(df)
  }

  private def loadAllIndexers(path: String): Map[String, StringIndexerModel] = {
    val indexerFiles = Files.list(Paths.get(path)).filter(Files.isDirectory(_)).iterator()
    var indexerMap: Map[String, StringIndexerModel] = Map()

    while (indexerFiles.hasNext) {
      val filePath = indexerFiles.next().toString
      val columnName = filePath.split("/").last
      if (Files.exists(Paths.get(filePath))) {
        indexerMap += (columnName -> StringIndexerModel.load(filePath))
      } else {
        throw new Exception(s"StringIndexer for $columnName not found at $filePath")
      }
    }
    indexerMap
  }

  private def loadAssembler(str: String): VectorAssembler = {
    if (Files.exists(Paths.get(str))) {
      VectorAssembler.load(str)
    } else {
      throw new Exception(s"VectorAssembler not found at $str")
    }
  }

  private def loadAllEncoders(path: String): Map[String, OneHotEncoderModel] = {
    val encodersFiles = Files.list(Paths.get(path)).filter(Files.isDirectory(_)).iterator()
    var encodersMap: Map[String, OneHotEncoderModel] = Map()

    while (encodersFiles.hasNext) {
      val filePath = encodersFiles.next().toString
      val columnName = filePath.split("/").last
      if (Files.exists(Paths.get(filePath))) {
        encodersMap += (columnName -> OneHotEncoderModel.load(filePath))
      } else {
        throw new Exception(s"StringIndexer for $columnName not found at $filePath")
      }
    }
    encodersMap
  }

  private def loadWord2VecModel(path: String): Word2VecModel = {
    if (Files.exists(Paths.get(path))) {
      Word2VecModel.load(path)
    } else {
      throw new Exception(s"Word2Vec model not found at $path")
    }
  }

  private def loadHashing(path: String): HashingTF = {
    if (Files.exists(Paths.get(path))) {
      HashingTF.load(path)
    } else {
      throw new Exception(s"HashingTF model not found at $path")
    }
  }

}

object FeaturePreprocessor {
  val modelsPath: String = "models/"
  val indexersPath: String = modelsPath + "indexers/"
  val assemblerPath: String = modelsPath + "assemblers/assembler"
  val word2VecPath: String = modelsPath + "word2vec/"
  val hashingEntityPath: String = modelsPath + "hashingEntityId/"

  def assembleData(trainData: DataFrame): DataFrame = {
    trainWord2VecModel(trainData)
      .transform(trainHashingEntityId)
      .transform(trainStringIndexing)
      .transform(trainAssembler)
  }

  private def trainWord2VecModel(trainData: DataFrame): DataFrame = {
    val tokenizedDeviceInfoDataTrain = trainData.withColumn("DeviceInfoTokens", split(col("DeviceInfo"), " "))

    if (Files.exists(Paths.get(word2VecPath))) {
      return Word2VecModel.load(word2VecPath).transform(tokenizedDeviceInfoDataTrain)
    }

    val word2VecDeviceInfo = new Word2Vec()
      .setInputCol("DeviceInfoTokens")
      .setOutputCol("DeviceInfoVector")
      .setVectorSize(20)
      .setMinCount(1)

    val word2VecModel = word2VecDeviceInfo.fit(tokenizedDeviceInfoDataTrain)

    word2VecModel.write.overwrite().save(word2VecPath);

    word2VecModel.transform(tokenizedDeviceInfoDataTrain)
  }

  def trainHashingEntityId(data: DataFrame): DataFrame = {
    // Tokenize the ENTITY_ID column
    val tokenizedData = data.withColumn("ENTITY_ID_tokens", split(col("ENTITY_ID"), "_"))

    // Apply HashingTF
    val hashingTF = new HashingTF()
      .setInputCol("ENTITY_ID_tokens")
      .setOutputCol("ENTITY_ID_hashed")
      .setNumFeatures(1000)

    hashingTF.write.overwrite().save(hashingEntityPath);

    hashingTF.transform(tokenizedData)
  }

  private def trainStringIndexing(df: DataFrame): DataFrame = {
    if (Files.exists(Paths.get(indexersPath))) {

      val indexerFiles = Files.list(Paths.get(indexersPath)).filter(Files.isDirectory(_)).iterator()
      var indexerMap: Map[String, StringIndexerModel] = Map()

      while (indexerFiles.hasNext) {
        val filePath = indexerFiles.next().toString
        val columnName = filePath.split("/").last
        if (Files.exists(Paths.get(filePath))) {
          indexerMap += (columnName -> StringIndexerModel.load(filePath))
        } else {
          throw new Exception(s"StringIndexer for $columnName not found at $filePath")
        }
      }
      indexerMap.foldLeft(df) { (data, indexer) =>
        indexer._2.transform(data)
      }
    }

    val stringColumnsToIndex = schema.fields
      .filter(f => f.dataType == StringType && f.name != "DeviceInfo")
      .filter(f => f.dataType == StringType && f.name != "ENTITY_ID")
      .map(_.name)

    val indexers = stringColumnsToIndex.map { colName =>
      new StringIndexer()
        .setInputCol(colName)
        .setHandleInvalid("skip")
        .setOutputCol(colName + "_indexed")
        .fit(df)
    }

    // Збереження кожного StringIndexerModel
    indexers.zip(stringColumnsToIndex).foreach { case (indexerModel, colName) =>
      val path = s"$indexersPath$colName"
      indexerModel.write.overwrite().save(path)
      println(s"Saved indexer for column $colName to $path")
    }

    indexers.foldLeft(df) { (df, indexer) =>
      indexer.transform(df)
    }
  }

  private def trainAssembler(df: DataFrame): DataFrame = {
    if (Files.exists(Paths.get(assemblerPath))) {
      return VectorAssembler.load(assemblerPath).transform(df)
    }

    val stringColumns = schema.fields.filter(f => f.dataType == StringType).map(_.name)
    val excludedColumns = stringColumns ++ Seq(
      "isFraud",
      "DeviceInfoTokens",
      "DeviceInfoVector",
      "ENTITY_ID_tokens",
      "P_emaildomain_indexed",
      "R_emaildomain_indexed",
      "label"
    )

    val featureColumns = df.columns.diff(excludedColumns)

    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    assembler.write.overwrite().save(assemblerPath)

    assembler.transform(df)
  }

  val schema: StructType = StructType(
    Seq(
      StructField("TransactionAmt", DoubleType, nullable = true),
      StructField("EVENT_TIMESTAMP", DoubleType, nullable = true),
      StructField("ENTITY_ID", StringType, nullable = true),
      StructField("ProductCD", StringType, nullable = true),
      StructField("card1", IntegerType, nullable = true),
      StructField("card2", DoubleType, nullable = true),
      StructField("card3", DoubleType, nullable = true),
      StructField("card5", DoubleType, nullable = true),
      StructField("card6", StringType, nullable = true),
      StructField("addr1", DoubleType, nullable = true),
      StructField("dist1", DoubleType, nullable = true),
      StructField("C1", DoubleType, nullable = true),
      StructField("C2", DoubleType, nullable = true),
      StructField("C4", DoubleType, nullable = true),
      StructField("C5", DoubleType, nullable = true),
      StructField("C6", DoubleType, nullable = true),
      StructField("C7", DoubleType, nullable = true),
      StructField("C8", DoubleType, nullable = true),
      StructField("C9", DoubleType, nullable = true),
      StructField("C10", DoubleType, nullable = true),
      StructField("C11", DoubleType, nullable = true),
      StructField("C12", DoubleType, nullable = true),
      StructField("C13", DoubleType, nullable = true),
      StructField("C14", DoubleType, nullable = true),
      StructField("V62", DoubleType, nullable = true),
      StructField("V70", DoubleType, nullable = true),
      StructField("V76", DoubleType, nullable = true),
      StructField("V78", DoubleType, nullable = true),
      StructField("V82", DoubleType, nullable = true),
      StructField("V91", DoubleType, nullable = true),
      StructField("V127", DoubleType, nullable = true),
      StructField("V130", DoubleType, nullable = true),
      StructField("V139", DoubleType, nullable = true),
      StructField("V160", DoubleType, nullable = true),
      StructField("V165", DoubleType, nullable = true),
      StructField("V187", DoubleType, nullable = true),
      StructField("V203", DoubleType, nullable = true),
      StructField("V207", DoubleType, nullable = true),
      StructField("V209", DoubleType, nullable = true),
      StructField("V210", DoubleType, nullable = true),
      StructField("V221", DoubleType, nullable = true),
      StructField("V234", DoubleType, nullable = true),
      StructField("V257", DoubleType, nullable = true),
      StructField("V258", DoubleType, nullable = true),
      StructField("V261", DoubleType, nullable = true),
      StructField("V264", DoubleType, nullable = true),
      StructField("V266", DoubleType, nullable = true),
      StructField("V267", DoubleType, nullable = true),
      StructField("V271", DoubleType, nullable = true),
      StructField("V274", DoubleType, nullable = true),
      StructField("V277", DoubleType, nullable = true),
      StructField("V283", DoubleType, nullable = true),
      StructField("V285", DoubleType, nullable = true),
      StructField("V289", DoubleType, nullable = true),
      StructField("V291", DoubleType, nullable = true),
      StructField("V294", DoubleType, nullable = true),
      StructField("id_01", DoubleType, nullable = true),
      StructField("id_02", DoubleType, nullable = true),
      StructField("id_05", DoubleType, nullable = true),
      StructField("id_06", DoubleType, nullable = true),
      StructField("id_09", DoubleType, nullable = true),
      StructField("id_13", DoubleType, nullable = true),
      StructField("id_17", DoubleType, nullable = true),
      StructField("id_19", DoubleType, nullable = true),
      StructField("id_20", DoubleType, nullable = true),
      StructField("DeviceType", StringType, nullable = true),
      StructField("DeviceInfo", StringType, nullable = true)
    )
  )

}
