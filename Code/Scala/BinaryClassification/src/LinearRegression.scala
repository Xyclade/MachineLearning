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
    val basePath = "/Users/mikedewaard/ML_for_Hackers/05-Regression/data/longevity.csv"

    val test_data = GetDataFromCSV(new File(basePath))


    val plot =  LinePlot.plot(test_data._1,Line.Style.SOLID,Color.red)//ScatterPlot.plot(test_data._1, test_data._2, '@', Array(Color.red, Color.blue));
     peer.setContentPane(plot)
    size = new Dimension(400, 400)

    //data - a matrix containing the explanatory variables.
    val data : Array[Array[Double]] =  Array(Array(50.0),Array(100.0),Array(150.0))


    //y - the response values.
      val response : Array[Double] = Array(100.0,200.0, 300.0)

    val ols = new OLS(data,response)
   val result =  ols.predict(Array(200.0))

println(result);

  }


  def GetDataFromCSV(file: File): (Array[Array[Double]], Array[Array[Double]]) = {
    val source = scala.io.Source.fromFile(file)
    val data = source.getLines().drop(1).map(x => GetDataFromString(x)).toArray
    source.close()
    val smokerAges = data.filter(x => x._1 == 1).map(y => y._2.toDouble);
    val nonSmokerAges = data.filter(x => x._1 == 0).map(y => y._2);

    val nonSmokerDensity = nonSmokerAges.groupBy(x => x).toArray.sortBy(x => x._1).map(y => Array(y._1.toDouble, y._2.length.toDouble / nonSmokerAges.length)).toArray
    val smokerDensity = smokerAges.groupBy(x => x).toArray.sortBy(x => x._1).map(y => Array(y._1.toDouble, y._2.length.toDouble / smokerAges.length)).toArray

    return (smokerDensity, nonSmokerDensity)

  }

  def GetDataFromString(dataString: String): (Int, Int) = {

    //Split the comma separated value string into an array of strings
    val dataArray: Array[String] = dataString.split(',')

    //Extract the values from the strings
    val smoker: Int = dataArray(0).toInt
    val ageAtDeath: Int = dataArray(1).toInt

    //And return the result in a format that can later easily be used to feed to Smile
    return (smoker, ageAtDeath)
  }
}