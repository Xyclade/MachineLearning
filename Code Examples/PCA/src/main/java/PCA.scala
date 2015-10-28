import java.awt.{Dimension, Color}
import java.io.{PrintWriter, File}
import java.text.{DecimalFormat, DateFormat, SimpleDateFormat}
import java.util.{Locale, Date}

import smile.math.distance.CorrelationDistance
import smile.plot.{PlotCanvas, LinePlot, ScatterPlot, Line}
import smile.projection.PCA

import scala.swing.{MainFrame, SimpleSwingApplication}
import scala.util.Random


object PCA extends SimpleSwingApplication{


  def top = new MainFrame {

     title = "PCA Example from http://xyclade.ml"
    //Get the example data
    val basePath = "data/"
    val exampleDataPath = basePath + "PCA_Example_1.csv"
    val trainData = GetStockDataFromCSV(new File(exampleDataPath))

    val pca = new PCA(trainData._2)

    //We want to merge into 1 feature
    pca.setProjection(1)
    val points = pca.project(trainData._2)

   val maxDataValue = points.maxBy(x => x(0))
   val minDataValue = points.minBy(x => x(0))
    val rangeValue = maxDataValue(0) - minDataValue(0)
    val plotData = points.zipWithIndex.map(x => Array(x._2.toDouble, -x._1(0) / rangeValue))
   // val plotData = points.zipWithIndex.map(x => Array(x._2.toDouble, x._1(0) ))
    val canvas: PlotCanvas = LinePlot.plot("Merged Features Index", plotData, Line.Style.DASH, Color.RED);


    //Verification against DJI
    val verificationDataPath = basePath + "PCA_Example_2.csv"
    val verificationData = GetDJIFromFile(new File(verificationDataPath))
    val DJIIndex = GetDJIFromFile(new File(verificationDataPath))
    canvas.line("Dow Jones Index", DJIIndex._2, Line.Style.DOT_DASH, Color.BLUE)


    peer.setContentPane(canvas)
    size = new Dimension(700, 400)

  }


  def GetStockDataFromCSV(file: File): (Array[Date],Array[Array[Double]]) = {
    val source = scala.io.Source.fromFile(file)
    //Get all the records (minus the header)
    val data = source.getLines().drop(1).map(x => GetStockDataFromString(x)).toArray
    source.close()
    //group all records by date, and sort the groups on date ascending
    val groupedByDate = data.groupBy(x => x._1).toArray.sortBy(x => x._1)
    //extract the values from the 3-tuple and turn them into an array of tuples: Array[(Date, Array[Double)]
    val dateArrayTuples = groupedByDate.map(x => (x._1, x._2.sortBy(x => x._2).map(y => y._3)))

    //turn the tuples into two separate arrays for easier use later on
    val dateArray = dateArrayTuples.map(x => x._1).toArray
    val doubleArray = dateArrayTuples.map(x => x._2).toArray


    (dateArray,doubleArray)
  }

  def GetStockDataFromString(dataString: String): (Date,String,Double) = {

    //Split the comma separated value string into an array of strings
    val dataArray: Array[String] = dataString.split(',')

    val format = new SimpleDateFormat("yyyy-MM-dd")
    //Extract the values from the strings

    val date = format.parse(dataArray(0))
    val stock: String = dataArray(1)
    val close: Double = dataArray(2).toDouble

    //And return the result in a format that can later easily be used to feed to Smile
    (date,stock,close)
  }



  def GetDJIRecordFromString(dataString: String): (Date,Double) = {

    //Split the comma separated value string into an array of strings
    val dataArray: Array[String] = dataString.split(',')

    val format = new SimpleDateFormat("yyyy-MM-dd")
    //Extract the values from the strings

    val date = format.parse(dataArray(0))
    val close: Double = dataArray(4).toDouble

    //And return the result in a format that can later easily be used to feed to Smile
    (date,close)
  }


  def GetDJIFromFile(file: File): (Array[Date],Array[Double]) = {
    val source = scala.io.Source.fromFile(file)
    //Get all the records (minus the header)
    val data = source.getLines().drop(1).map(x => GetDJIRecordFromString(x)).toArray
    source.close()

    //turn the tuples into two separate arrays for easier use later on
    val sortedData = data.sortBy(x => x._1)
    val dates = sortedData.map(x => x._1)
    val maxDouble = sortedData.maxBy(x => x._2)._2
    val minDouble = sortedData.minBy(x => x._2)._2
    val rangeValue = maxDouble - minDouble
    val doubles = sortedData.map(x =>   x._2 / rangeValue )



    (dates, doubles)
  }
}