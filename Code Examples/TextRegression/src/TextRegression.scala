import java.awt.Color
import java.io.File
import com.github.tototoshi.csv._
import smile.plot._

import scala.swing._

/**
 * Created by mikedewaard on 17/02/15.
 */
object TextRegression extends SimpleSwingApplication {

    def top = new MainFrame {
      title = "Text regression Example"
      val basePath = "/users/mikedewaard/MachineLearning/Example Data/TextRegression_Example_4.csv"

      val test_data = GetDataFromCSV(new File(basePath))

      //   val plot = ScatterPlot.plot(test_data._1, test_data._2, '@', Array(Color.red, Color.blue))
      //   peer.setContentPane(plot)
      size = new Dimension(400, 400)
      val documentTDM = new TDM();
      test_data.foreach(x => x._2.split(" ").filter(_.nonEmpty).foreach(y => documentTDM.addTermToRecord(y, x._1)))
      val StopWords = getStopWords()
      documentTDM.records = documentTDM.records.filter(x => !StopWords.contains(x.term))
      documentTDM.SortByTotalFrequency
      val hackerRecord = documentTDM.records.find(x => x.term == "hacker")
      val records = documentTDM.records

    }

  def GetDataFromCSV(file: File) : List[(String,String)]= {
    val reader = CSVReader.open(file)
    val data = reader.all()

    val documents = data.drop(1).map(x => (x(1),x(4)))
    return documents
  }

  def GetDataFromString(dataString: String): (String, String) = {

    //Split the comma separated value string into an array of strings
    val dataArray: Array[String] = dataString.split(',q')

    //Extract the values from the strings
    val title: String = dataArray(1)
    val longDescription : String = dataArray(4).toLowerCase()


    //And return the result in a format that can later easily be used to feed to Smile
    return (title,longDescription)
  }

  def getStopWords(): List[String] = {
    val source = scala.io.Source.fromFile(new File("/Users/mikedewaard/MachineLearning/Example Data/stopwords.txt"))("latin1")
    val lines = source.mkString.split("\n")
    source.close()
    return lines.toList
  }

}
