import java.awt.{Color, Dimension}
import java.io.{ File}
import java.text.DecimalFormat

import smile.math.kernel.{ GaussianKernel}
import smile.plot.ScatterPlot
import smile.classification.SVM
import scala.collection.mutable
import scala.swing.{SimpleSwingApplication, MainFrame}

object SupportVectorMachine extends SimpleSwingApplication {


  def top = new MainFrame {
    title = "SVM Examples"
    //File path (this changes per example)
    val path =  "/users/mikedewaard/MachineLearning/Example Data/SVM_Example_1.csv"
    val df = new DecimalFormat("#.#")
    //Loading of the test data and plot generation stays the same
    val testData = GetDataFromCSV(new File(path))
    val plot = ScatterPlot.plot(testData._1, testData._2, '@', Array(Color.blue, Color.green))
    peer.setContentPane(plot)

    //Here we do our SVM fine tuning with possibly different kernels
    //val svm = new SVM[Array[Double]](new GaussianKernel(0.01), 10.0,2)
    val printlist = mutable.MutableList[(Double,Double,Double)]()



    val avgYSquared = Math.pow(testData._1.map(x => x(1)).sum / testData._1.length, 2)
    val avgSquaredY = testData._1.map(x => Math.pow(x(1),2)).sum / testData._1.length

    val varianceY = avgSquaredY - avgYSquared
    val sigmaY = Math.sqrt(varianceY)
    println("VarianceY: " + varianceY)
    println("sigmaY: " + sigmaY)

    val avgXSquared = Math.pow(testData._1.map(x => x(0)).sum / testData._1.length, 2)
    val avgSquaredX = testData._1.map(x => Math.pow(x(0),2)).sum / testData._1.length

    val varianceX = avgSquaredX - avgXSquared
    val sigmaX = Math.sqrt(varianceX)
    println("VarianceX: " + varianceX)
    println("sigmaX: " + sigmaX)

    val sigmaXY = sigmaX + sigmaY
    val sigmaXYAVG =  sigmaXY/2

    val sigmas = Array(0.001,0.01,0.1,0.2,0.5,1.0,2.0,3.0,10.0,100, sigmaX, sigmaY,sigmaXY,sigmaXYAVG)
    val marginPenalties = Array(0.001,0.01,0.1,0.2,0.5,1.0,2.0,3.0,10.0,100)



    sigmas.foreach( sigma =>
      marginPenalties.foreach(marginPenalty => {
      val svm = new SVM[Array[Double]](new GaussianKernel(sigma), marginPenalty, 2)
      svm.learn(testData._1, testData._2)
      svm.finish()


      //Calculate how well the SVM predicts on the training set
      val predictions = testData._1.map(x => svm.predict(x)).zip(testData._2)
      val falsePredictions = predictions.map(x => if (x._1 == x._2) 0 else 1)

      val result = (sigma ,marginPenalty , (falsePredictions.sum.toDouble / predictions.length * 100))
      printlist += result
        println("sigma: " + sigma + " margin: " + marginPenalty + " error: " + result._3)
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
