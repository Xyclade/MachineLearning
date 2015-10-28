import java.awt.{Color, Dimension}
import java.io.{ File}
import java.text.DecimalFormat

import smile.math.kernel.{PolynomialKernel, GaussianKernel}
import smile.plot.ScatterPlot
import smile.classification.SVM
import scala.collection.mutable
import scala.swing.{SimpleSwingApplication, MainFrame}

object SVM_Example_2 extends SimpleSwingApplication {


  def top = new MainFrame {
    title = "SVM Example 2"
    //File path (this changes per example)
    val trainingPath =  "data/SVM_Example_2.csv"
    val testingPath = "data/SVM_Example_2_Test_data.csv"
    val df = new DecimalFormat("#.#")
    //Loading of the test data and plot generation stays the same
    val trainingData = GetDataFromCSV(new File(trainingPath))
    val testingData  = GetDataFromCSV(new File(testingPath))


    val plot = ScatterPlot.plot(trainingData._1, trainingData._2, '@', Array(Color.blue, Color.green))
    peer.setContentPane(plot)

    val printlist = mutable.MutableList[(Int,Double,Double)]()


    val sigmas = Array(2,3,4,5)
    val marginPenalties = Array(0.001,0.01,0.1,0.2,0.5,1.0,2.0,3.0,10.0,100)



    sigmas.foreach( sigma =>
      marginPenalties.foreach(marginPenalty => {
        val svm = new SVM[Array[Double]](new PolynomialKernel(sigma), marginPenalty, 2)
        svm.learn(trainingData._1, trainingData._2)
        svm.finish()


        //Calculate how well the SVM predicts on the training set
        val predictions = testingData._1.map(x => svm.predict(x)).zip(testingData._2)
        val falsePredictions = predictions.map(x => if (x._1 == x._2) 0 else 1)

        val result = (sigma ,marginPenalty , (falsePredictions.sum.toDouble / predictions.length * 100))
        printlist += result
        println("degree: " + sigma + " margin: " + marginPenalty + " error: " + result._3)
      }
      )
    )

    print("| |")
    sigmas.foreach(x => print(" s: " + x + " |"))
    println("")
    println("| :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | ")
    marginPenalties.foreach(x => {
      val sigmaValues =  sigmas.map(y => printlist.filter(z => z._1 == y && z._2 == x)(0)._3)
      println("")
      print("| **c: " + x + "** |")
      sigmaValues.foreach(s => print(" " + df.format(s) + "% |")
      )
    })



    size = new Dimension(400, 400)
  }


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
    val coordinates  = Array( dataArray(0).toDouble, dataArray(1).toDouble)
    val classifier: Int = dataArray(2).toInt

    //And return the result in a format that can later easily be used to feed to Smile
    return (coordinates, classifier)
  }
}
