
import java.awt.Color
import java.io.File

import smile.classification.KNN
import smile.plot.ScatterPlot
import smile.validation.CrossValidation

import scala.swing._
import scala.swing.event.ButtonClicked

/**
 * Created by mikedewaard on 17/12/14.
 */
object KNNExample{ /*extends SimpleSwingApplication {
  def top = new MainFrame {
    title = "KNN Example!"
    val basePath = "/Users/mikedewaard/ML_for_Hackers/10-Recommendations/data/example_data.csv"


    val test_data = GetDataFromCSV(new File(basePath))

    val plot = ScatterPlot.plot(test_data._1, test_data._2, '@', Array(Color.red, Color.blue));
    peer.setContentPane(plot)
    size = new Dimension(400, 400)

  }
*/
/*
  def main(args: Array[String]): Unit = {
    val basePath = "/Users/mikedewaard/ML_for_Hackers/10-Recommendations/data/example_data.csv"
    val testData = GetDataFromCSV(new File(basePath))

    //Define the amount of rounds, in our case 2 and initialize the cross validation
    val validationRounds = 2;
    val cv = new CrossValidation(testData._2.length, validationRounds);
    //Then for each round
    for (i <- 0 to validationRounds - 1) {

      //Generate a subset of data points and their classifiers for Training
      val dpForTraining = testData._1.zipWithIndex.filter(x => cv.test(i).toList.contains(x._2)).map(y => y._1);
      val classifiersForTraining = testData._2.zipWithIndex.filter(x => cv.test(i).toList.contains(x._2)).map(y => y._1);

      //And the corresponding subset of datapoints and their classifiers for testing
      val dpForTesting = testData._1.zipWithIndex.filter(x => !cv.test(i).contains(x._2)).map(y => y._1);
      val classifiersForTesting = testData._2.zipWithIndex.filter(x => !cv.test(i).contains(x._2)).map(y => y._1);

      //Then generate a Model with KNN
      val knn = KNN.learn(dpForTraining, classifiersForTraining, 3);

      //And for each test data point make a prediction with the model
      val predictions = dpForTesting.map(x => knn.predict(x))

      //Finally evaluate the predictions as correct or incorrect and count the amount of wrongly classified data points.
      var error = 0.0;
      for (j <- 0 to predictions.length - 1) {
        if (predictions(j) != classifiersForTesting(j)) {
          error += 1
        }
      }
      println("false prediction rate: " +  error / predictions.length * 100 + "%")

      //Use the model for new predictions:
      val unknownDatapoint = Array(4.5,7.5)

      val result = knn.predict(unknownDatapoint);

      if (result == 0)
      {
        println("Internet Service Provider Alpha")
      }
      else if (result == 1)
      {
        println("Internet Service Provider Beta")
      }
      else
      {
        println("Unexpected prediction")
      }
    }
  }*/

  def GetDataFromCSV(file: File): (Array[Array[Double]], Array[Int]) = {
    val source = scala.io.Source.fromFile(file)
    val data = source.getLines().drop(1).map(x => GetDataFromString(x)).toArray
    source.close()
    val dataPoints = data.map(x => x._1)
    val classifierArray = data.map(x => x._2)
    return (dataPoints, classifierArray)

  }

  def GetDataFromString(dataString: String): (Array[Double], Int) = {

    //Split the comma separated value string into an array of strings
    val dataArray: Array[String] = dataString.split(',')

    //Extract the values from the strings
    val xCoordinate: Double = dataArray(0).toDouble
    val yCoordinate: Double = dataArray(1).toDouble
    val classifier: Int = dataArray(2).toInt

    //And return the result in a format that can later easily be used to feed to Smile
    return (Array(xCoordinate, yCoordinate), classifier)
  }
}