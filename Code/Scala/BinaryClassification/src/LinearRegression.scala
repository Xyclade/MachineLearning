import java.awt.Color
import java.io.File

import smile.plot._
import smile.regression._
import scala.swing._

/**
 * Created by mikedewaard on 04/02/15.
 */

object LinearRegression extends SimpleSwingApplication {
  def top = new MainFrame {
    title = "KNN Example!"
    val basePath = "/Users/mikedewaard/MachineLearning/Example Data/OLS_Regression_Example_3.csv"

    val test_data = GetDataFromCSV(new File(basePath))

val plotData = (test_data._1 zip test_data._2).map(x => Array(x._1(1) ,x._2))
val maleFemaleLabels = test_data._1.map( x=> x(0).toInt);
    val plot =  ScatterPlot.plot(plotData,maleFemaleLabels,'@',Array(Color.red, Color.blue, Color.green))
     peer.setContentPane(plot)
    size = new Dimension(400, 400)

    val ols = new OLS(test_data._1,test_data._2)

    println("Error: " + ols.error())
    println(ols.predict(Array(1.0,60.0)));
println(ols.predict(Array(2.0,60.0)));

  }


  def GetDataFromCSV(file: File): (Array[Array[Double]], Array[Double]) = {
    val source = scala.io.Source.fromFile(file)
    val data = source.getLines().drop(1).map(x => GetDataFromString(x)).toArray
    source.close()
    var inputData = data.map(x => x._1)
    var resultData = data.map(x => x._2)

    return (inputData,resultData)
  }

  def GetDataFromString(dataString: String): (Array[Double], Double) = {

    //Split the comma separated value string into an array of strings
    val dataArray: Array[String] = dataString.split(',')
    var person = 2.0;

 if (dataArray(0) == "\"Male\"") {
   person = 1.0
 }

    //Extract the values from the strings
    val data : Array[Double] = Array(person,dataArray(1).toDouble)
    val weight: Double = dataArray(2).toDouble

    //And return the result in a format that can later easily be used to feed to Smile
    return (data, weight)
  }
}