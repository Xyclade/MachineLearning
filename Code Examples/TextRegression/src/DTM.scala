import java.io.File
import scala.collection.mutable

class DTM {

  val records: mutable.MutableList[DTMRecord] = mutable.MutableList[DTMRecord]()
  val wordList: mutable.Set[String] = mutable.Set[String]()
  val stopWords = getStopWords

  def addDocumentToRecords(documentName: String, rank: Int, documentContent: String) = {

    val wordRecords = mutable.HashMap[String, Int]()
    val filteredDocumentWords  = documentContent.toLowerCase.split(" ").filter(x => !stopWords.contains(x))

    filteredDocumentWords.foreach { x =>
        val wordRecord = wordRecords.collectFirst( {case (`x`,amount) => (x,amount + 1) }).getOrElse(x -> 1)
          wordRecords += wordRecord
          wordList += x
    }
    records += new DTMRecord(documentName, rank, wordRecords)
  }

  def getStopWords: List[String] = {
    val source = scala.io.Source.fromFile(new File("/Users/mikedewaard/MachineLearning/Example Data/stopwords.txt"))("latin1")
    val lines = source.mkString.split("\n")
    source.close()
    lines.toList
  }

  def getNumericRepresentationForRecords: (Array[Array[Double]], Array[Double]) = {
    val dtmNumeric = mutable.MutableList[Array[Double]]()
    val ranks = mutable.MutableList[Double]()

    records.foreach { record =>
      //Add the rank to the array of ranks
      ranks += record.rank.toDouble
      //And create an array representing all words and their occurrences for this document:
      val dtmNumericRecord = wordList
        .map(word =>  record.occurrences
        .collectFirst{ case (`word`,amount) => amount.toDouble}
        .getOrElse(0.0)).toArray

        dtmNumeric += dtmNumericRecord
    }
    (dtmNumeric.toArray, ranks.toArray)
  }
}

class DTMRecord(val document : String, val rank : Int, var occurrences :  mutable.HashMap[String,Int] )

