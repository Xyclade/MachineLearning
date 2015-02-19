import java.awt.Color
import java.io.File
import java.util.Calendar
import com.github.tototoshi.csv._
import smile.plot._
import smile.regression.{LASSO, RidgeRegression}
import smile.validation.CrossValidation

import scala.swing._

/**
 * Created by mikedewaard on 17/02/15.
 */
object TextRegression extends SimpleSwingApplication {

    def top = new MainFrame {
      title = "Text regression Example"
      val basePath = "/users/mikedewaard/MachineLearning/Example Data/TextRegression_Example_4.csv"
      println("\t\t\t\t starting")
      val testData = GetDataFromCSV(new File(basePath))

      //   val plot = ScatterPlot.plot(test_data._1, test_data._2, '@', Array(Color.red, Color.blue))
      //   peer.setContentPane(plot)
      size = new Dimension(400, 400)
      val documentTermMatrix  = new DTM()
      testData.foreach(x => documentTermMatrix.addDocumentToRecords(x._1,x._2,x._3))


      val cv = new CrossValidation(testData.length, 2)
      val numericDTM = documentTermMatrix.getNumericRepresentationForRecords()
      println(Calendar.getInstance().getTime() + " Finished getting the matrix representation of the DTM")
      for (i <-0 until cv.k) {
        println(Calendar.getInstance().getTime() +" Starting round:" + i )
        val dpForTraining = numericDTM._1.zipWithIndex.filter(x => cv.test(i).toList.contains(x._2)).map(y => y._1)
        val classifiersForTraining = numericDTM._2.zipWithIndex.filter(x => cv.test(i).toList.contains(x._2)).map(y => y._1)

        //And the corresponding subset of data points and their classifiers for testing
        val dpForTesting = numericDTM._1.zipWithIndex.filter(x => !cv.test(i).contains(x._2)).map(y => y._1)
        val classifiersForTesting = numericDTM._2.zipWithIndex.filter(x => !cv.test(i).contains(x._2)).map(y => y._1)
        println(Calendar.getInstance().getTime() +" Separated the training and testing set for round:" + i )

        val lambdas: Array[Double] = Array(0.1, 0.25, 0.5, 1.0, 2.0, 5.0)

        lambdas.foreach { x =>
          println(Calendar.getInstance().getTime() +" Running lamda: " + x  + " for round:" + i )
        val model = new LASSO(dpForTraining,classifiersForTraining,x)
          println(Calendar.getInstance().getTime() +" Trained the model for lamda: " + x  + " for round:" + i )
        val results  =  dpForTesting.map(y => model.predict(y)) zip classifiersForTesting
          println(Calendar.getInstance().getTime() +" Got results for  lamda: " + x  + " for round:" + i )
        val rmse =  Math.sqrt(results.map(x => Math.pow(x._1 - x._2,2)).sum / results.length)
        println(Calendar.getInstance().getTime() +"\t\t\t\t\t\t\t\t\t\tLambda: " + x + " RMSE: " + rmse);


        }
      }
    }

  def GetDataFromCSV(file: File) : List[(String,Int,String)]= {
    val reader = CSVReader.open(file)
    val data = reader.all()

    val documents = data.drop(1).map(x => (x(1),x(3)toInt,x(4)))
    return documents
  }





}
