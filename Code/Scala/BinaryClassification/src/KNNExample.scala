import java.awt.Color
import java.io.File
import javax.swing.JPanel

import smile.classification.KNN
import smile.plot.{PlotCanvas, ScatterPlot}
import smile.validation.CrossValidation

/**
 * Created by mikedewaard on 15/01/15.
 */
class KNNExamplex {


  def getPlot  : PlotCanvas  =
  {
    val basePath = "/Users/mikedewaard/ML_for_Hackers/10-Recommendations/data/example_data.csv"


    val test_data = GetDataFromCSV(new File(basePath))

    val plot =   ScatterPlot.plot(test_data._1,test_data._2,'@', Array(Color.red,Color.blue));
    return plot
  }

  //Values from: http://people.revoledu.com/kardi/tutorial/KNN/KNN_Numerical-example.html

 /* def main(args: Array[String]): Unit = {


    val basePath = "/Users/mikedewaard/ML_for_Hackers/10-Recommendations/data/example_data.csv"


    val test_data = GetDataFromCSV(new File(basePath))

    val plot =  ScatterPlot.plot(test_data._1, Color.red);

    val validationRounds = 2;
    val n = test_data._1.length;
    val cv  = new CrossValidation(100,validationRounds);

    val subsetofDPForTraining = test_data._1.zipWithIndex .filter( x=> cv.test(0).toList.contains(x._2) ).map(y => y._1);
    val subsetofDPClassifiersForTraining = test_data._2.zipWithIndex.filter( x=> cv.test(0).toList.contains(x._2) ).map(y => y._1);

    val subsetofDPForTesting = test_data._1.zipWithIndex.filter( x=> !cv.test(0).contains(x._2) ).map(y => y._1);
    val subsetofDPClassifiersForTesting = test_data._2.zipWithIndex.filter( x=> !cv.test(0).contains(x._2) ).map(y => y._1);


    for (i <- 0 to  validationRounds) {

      val knn = KNN.learn(subsetofDPForTraining, subsetofDPClassifiersForTraining, 3);


      val predictions = subsetofDPForTesting.map(x => knn.predict(x))
      var error = 0;
      for ( j <- 0 to predictions.length -1)
      {
        if (predictions(j) != subsetofDPClassifiersForTesting(j))
        {
          error +=1
        }

      }
      println("Error: " + error )

      val trainingIndexes : Array[Array[Int]] = cv.train
      val testingIndexes : Array[Array[Int]] = cv.test

      val trainingData = smile.math.Math.slice(test_data._1, trainingIndexes(0))
      val testData = smile.math.Math.slice(test_data._1, trainingIndexes(0))

    }
  }*/

  def GetDataFromCSV(file: File): (Array[Array[Double]], Array[Int]) = {
    // Get a buffered source
    val source = scala.io.Source.fromFile(file)
    //Get all lines from the csv, and drop the header
    val data = source.getLines().drop(1).map(x => GetDataFromString(x)).toArray
    source.close()
    //Split the tuples split the datapoints and their classifier data up into two arrays to correspond to the requested input for the Smile KNN implementation
    val dataPoints = data.map(x => x._1)
    val classifierArray = data.map(x => x._2)
    //Find the first line break in the email, as this indicates the message body
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
