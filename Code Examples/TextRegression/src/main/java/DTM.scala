import java.io.File
import scala.collection.mutable

class DTM {

  var records: List[DTMRecord] = List[DTMRecord]()
  var wordList: List[String] = List[String]()

  def addDocumentToRecords(documentName: String, rank: Int, documentContent: String) = {
    //Find a record for the document
    val record = records.find(x => x.document == documentName)
    if (record.nonEmpty) {
      throw new Exception("Document already exists in the records")
    }

    var wordRecords = mutable.HashMap[String, Int]()
    val individualWords = documentContent.toLowerCase.split(" ")
    individualWords.foreach { x =>
      val wordRecord = wordRecords.find(y => y._1 == x)
      if (wordRecord.nonEmpty) {
        wordRecords += x -> (wordRecord.get._2 + 1)
      }
      else {
        wordRecords += x -> 1
        wordList = x :: wordList
      }
    }
    records = new DTMRecord(documentName, rank, wordRecords) :: records
  }

  def getStopWords(): List[String] = {
    val source = scala.io.Source.fromFile(new File("data/stopwords.txt"))("latin1")
    val lines = source.mkString.split("\n")
    source.close()
    return lines.toList
  }

  def getNumericRepresentationForRecords(): (Array[Array[Double]], Array[Double]) = {
    //First filter out all stop words:
    val StopWords = getStopWords()
    wordList = wordList.filter(x => !StopWords.contains(x))

    var dtmNumeric = Array[Array[Double]]()
    var ranks = Array[Double]()

    records.foreach { x =>
      //Add the rank to the array of ranks
      ranks = ranks :+ x.rank.toDouble

      //And create an array representing all words and their occurrences 
      //for this document:
      var dtmNumericRecord: Array[Double] = Array()
      wordList.foreach { y =>

        val termRecord = x.occurrences.find(z => z._1 == y)
        if (termRecord.nonEmpty) {
          dtmNumericRecord = dtmNumericRecord :+ termRecord.get._2.toDouble
        }
        else {
          dtmNumericRecord = dtmNumericRecord :+ 0.0
        }
      }
      dtmNumeric = dtmNumeric :+ dtmNumericRecord

    }

    return (dtmNumeric, ranks)
  }
}

class DTMRecord(val document : String,
                val rank : Int,
                var occurrences :  mutable.HashMap[String,Int]
                 )