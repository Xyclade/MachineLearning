import java.awt.{Dimension, Color}
import java.io.File
import java.text.{DateFormat, SimpleDateFormat}
import java.util.{Locale, Date}

import smile.math.distance.CorrelationDistance
import smile.plot.{LinePlot, ScatterPlot}
import smile.projection.PCA
import smile.plot.Line

import scala.swing.{MainFrame, SimpleSwingApplication}


object PCA extends SimpleSwingApplication{


  def top = new MainFrame {
    title = "PCA Example"
    //Get the example data
    val basePath = "/users/mikedewaard/MachineLearning/Example Data/"
    val exampleDataPath = basePath + "PCA_Example_1.csv"
    val verificationDataPath = basePath + "PCA_Example_2.csv"

    val testData = GetStockDataFromCSV(new File(exampleDataPath))

    val verificationData = GetDJIFromFile(new File(verificationDataPath))


    val pca  = new PCA(testData._2)

    val correlationComputer = new CorrelationDistance()

    var CorrelationsBetweenColumns : Array[(Int,Int,Double)]= Array()
    //For each column in the matrix
    for (i <-0 until testData._2(0).length)
    {

      //Get the complete column
      val firstColumn = testData._2.map(x => x(i))

      //And loop through the not yet computed columns
      for (j <- i + 1 until testData._2(0).length)
      {
        val secondColumn = testData._2.map(x => x(j))
        CorrelationsBetweenColumns =  CorrelationsBetweenColumns :+  (i,j, -correlationComputer.d(firstColumn, secondColumn) + 1)
      }

    }
    CorrelationsBetweenColumns = CorrelationsBetweenColumns.sortBy(x => x._3)
 //   CorrelationsBetweenColumns.zipWithIndex.foreach(x => println(x._2 + ": " + x._1))


    val rSquareds = CorrelationsBetweenColumns.map(x => (x._1,x._2, Math.pow(x._3,2))).sortBy(x => x._3)

    rSquareds.foreach(x => println(x._1 + ";" + x._2 + " = " + x._3))




    pca.setProjection(1)
    val projection = pca.getProjection
    val variances = pca.getVariance
    val cumValProportion = pca.getCumulativeVarianceProportion
    val loadings = pca.getLoadings
    val points = pca.project(testData._2)

    val plotData = points.zipWithIndex.map( x=> Array(x._2.toDouble, x._1(0)))



    val center = pca.getCenter
    println("Loaded data")

    val plot =   LinePlot.plot(plotData, Line.Style.SOLID)
    plot.add( LinePlot.plot(Array(Array(10.0,20.0),Array(11.0,45.0)),Line.Style.DASH))
    val plotData2 = points.zipWithIndex.map( x=> Array(x._2.toDouble + 10, x._1(0) + 10))
    val plot2 =   LinePlot.plot(plotData2, Line.Style.SOLID)



    peer.setContentPane(plot)
   size = new Dimension(400, 400)
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
    val doubles = sortedData.map(x => x._2)


    (dates, doubles)
  }
}
