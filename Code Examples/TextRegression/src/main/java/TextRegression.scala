import java.awt.Color
import java.io.File
import java.util.Calendar
import com.github.tototoshi.csv._
import smile.plot._
import smile.regression.{LASSO, RidgeRegression}
import smile.validation.CrossValidation
object TextRegression  {

  def main(args: Array[String]): Unit = {


    //Get the example data
    val basePath = "data/TextRegression_Example_1.csv"
    val testData = GetDataFromCSV(new File(basePath))

    //Create a document term matrix for the data
    val documentTermMatrix = new DTM()
    testData.foreach(x => documentTermMatrix.addDocumentToRecords(x._1, x._2, x._3))

    //Get the cross validation data
    val cv = new CrossValidation(testData.length, 2)
    val numericDTM = documentTermMatrix.getNumericRepresentationForRecords

    for (i <- 0 until cv.k) {
      //Split off the training datapoints and classifiers from the dataset
      val dpForTraining = numericDTM._1.zipWithIndex.filter(x => cv.test(i).toList.contains(x._2)).map(y => y._1)
      val classifiersForTraining = numericDTM._2.zipWithIndex.filter(x => cv.test(i).toList.contains(x._2)).map(y => y._1)

      //And the corresponding subset of data points and their classifiers for testing
      val dpForTesting = numericDTM._1.zipWithIndex.filter(x => !cv.test(i).contains(x._2)).map(y => y._1)
      val classifiersForTesting = numericDTM._2.zipWithIndex.filter(x => !cv.test(i).contains(x._2)).map(y => y._1)

      //These are the lambda values we will verify against
      val lambdas: Array[Double] = Array(0.1, 0.25, 0.5, 1.0, 2.0, 5.0)

      lambdas.foreach { x =>
        //Define a new model based on the training data and one of the lambda's
        val model = new LASSO(dpForTraining, classifiersForTraining, x)

        //Compute the RMSE for this model with this lambda
        val results = dpForTesting.map(y => model.predict(y)) zip classifiersForTesting
        val RMSE = Math.sqrt(results.map(x => Math.pow(x._1 - x._2, 2)).sum / results.length)
        println(Calendar.getInstance().getTime + "Lambda: " + x + " RMSE: " + RMSE)

      }
    }
  }

  def GetDataFromCSV(file: File) : List[(String,Int,String)]= {
    val reader = CSVReader.open(file)
    val data = reader.all()

    val documents = data.drop(1).map(x => (x(1),x(3).toInt,x(4)))
    documents
  }





}
