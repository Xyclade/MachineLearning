import java.awt.Color
import java.io.File

import smile.plot._
import smile.regression._
import scala.swing._


object LinearRegression extends SimpleSwingApplication {
  def top = new MainFrame {
    title = "Linear regression Example from http://Xyclade.ml"
    val basePath = "data/OLS_Regression_Example_3.csv"

    val test_data = GetDataFromCSV(new File(basePath))

    val plotData = (test_data._1 zip test_data._2).map(x => Array(x._1(1) ,x._2))
    val maleFemaleLabels = test_data._1.map( x=> x(0).toInt)
    val plot =  ScatterPlot.plot(plotData,maleFemaleLabels,'@',Array(Color.blue, Color.green))
    plot.setTitle("Weight and heights for males and females")
    plot.setAxisLabel(0,"Heights")
    plot.setAxisLabel(1,"Weights")



    peer.setContentPane(plot)
    size = new Dimension(400, 400)

    val olsModel = new OLS(test_data._1,test_data._2)

    println("Prediction for Male of 1.7M: " +olsModel.predict(Array(0.0,170.0)))
    println("Prediction for Female of 1.7M:" + olsModel.predict(Array(1.0,170.0)))

    println("Model Error:" + olsModel.error())
    println("Accuracy of the model: "  + olsModel.RSquared() * 100 + "%")
  }

  def GetDataFromCSV(file: File): (Array[Array[Double]], Array[Double]) = {
    val source = scala.io.Source.fromFile(file)
    val data = source.getLines().drop(1).map(x => GetDataFromString(x)).toArray
    source.close()
    var inputData = data.map(x => x._1)
    var resultData = data.map(x => x._2)

    (inputData,resultData)
  }

  def GetDataFromString(dataString: String): (Array[Double], Double) = {

    //Split the comma separated value string into an array of strings
    val dataArray: Array[String] = dataString.split(',')
    var person = 1.0

    if (dataArray(0) == "\"Male\"") {
      person = 0.0
    }

    //Extract the values from the strings
    //Since the data is in US metrics (inch and pounds we will recalculate this to cm and kilo's)
    val data : Array[Double] = Array(person,dataArray(1).toDouble * 2.54)
    val weight: Double = dataArray(2).toDouble * 0.45359237

    //And return the result in a format that can later easily be used to feed to Smile
    (data, weight)
  }
}