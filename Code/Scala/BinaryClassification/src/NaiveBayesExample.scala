import java.io.File

import smile.classification.NaiveBayes
import smile.feature.Bag

/**
 * Created by mikedewaard on 26/01/15.
 */
object NaiveBayesExample {

/*
def main(args: Array[String]): Unit = {
    val basePath = "/Users/mikedewaard/ML_for_Hackers/03-Classification/data"
    val spamPath = basePath + "/spam"
    val spam2Path = basePath + "/spam_2"
    val easyHamPath = basePath + "/easy_ham"
    val easyHam2Path = basePath + "/easy_ham_2"
    val hardHamPath = basePath + "/hard_ham"
    val hardHam2Path = basePath + "/hard_ham_2"

  val  amountOfSamplesPerSet = 500;

  val amountOfFeaturesToTake = 400;

  //First get a subset of the filenames for the spam sample set (500 is the complete set in this case)
  val listOfSpamFiles =   getFilesFromDir(spamPath).take(amountOfSamplesPerSet)
  //Then get the messages that are contained in these files
  val spamMails = listOfSpamFiles.map{x => (x,getMessage(x)) }
  //Then its time for feature selection, but in order to pick good features we have to gain more insight
  val spamTDM = new TDM();
  //Build up the Term-Document Matrix for spam emails
  spamMails.foreach(x => x._2.split(" ").filter(_.nonEmpty).foreach(y => spamTDM.addTermToRecord(y,x._1.getName)))
  //Sort the spam by total frequency for ease
  spamTDM.SortByOccurrenceRate(spamMails.size)
// println("SPAM with stopwords")
// spamTDM.records.take(50).foreach({x => for (i <- 0 to (x.occurrenceRate(500) * 100).toInt) {print(x.term +" " )}; println()});
  //Get the stopwords
  val StopWords = getStopWords()

  //Filter out all stopwords
  spamTDM.records = spamTDM.records.filter(x => !StopWords.contains(x.term));
//  println("SPAM without stopwords")
//  spamTDM.records.take(50).foreach({x => for (i <- 0 to (x.occurrenceRate(500) * 100).toInt) {print(x.term +" " )}; println()});
  //Take a subset of all non-stopwords found in the spam mail, where they are sorted on highest frequency first
  val spamFeatures = spamTDM.records.take(amountOfFeaturesToTake).map(x => x.term)



  //Get a subset of the filenames from the ham sampleset (note that in this case it is not neccesary to randomly sample as the emails are already randomly ordered)
  val listOfHamFiles =   getFilesFromDir(easyHamPath).take(amountOfSamplesPerSet)
  //Get the messages that are contained in the ham files
  val hamMails  = listOfHamFiles.map{x => (x,getMessage(x)) }
  //Then its time for feature selection specifically for the Ham messages, but in order to pick good features we have to gain more insight
  val hamTDM = new TDM();
  //Build up the Term-Document Matrix for ham emails
  hamMails.foreach(x => x._2.split(" ").filter(_.nonEmpty).foreach(y => hamTDM.addTermToRecord(y,x._1.getName)))
  //Sort the ham by total frequency for ease
  hamTDM.SortByOccurrenceRate(hamMails.size)
  //Filter out all stopwords
  hamTDM.records = hamTDM.records.filter(x => !StopWords.contains(x.term));
  //Take a subset of all non-stopwords found in the ham mail, where they are sorted on highest frequency first
  val hamFeatures = hamTDM.records.take(amountOfFeaturesToTake).map(x => x.term)

  //Now we have a set of ham and spam features, we group them and then remove the intersecting features, as these are noise.
  var data = (hamFeatures ++ spamFeatures).toSet
  hamFeatures.intersect(spamFeatures).foreach(x => data = (data - x))


  //Initialize a bag of words that takes the top x features from both spam and ham and combines them
  var bag = new Bag[String] (data.toArray);

  //Initialize the classifier array with first a set of 0(spam) and then a set of 1(ham) values that represent the emails
  var classifiers =  Array.fill[Int](amountOfSamplesPerSet)(0) ++  Array.fill[Int](amountOfSamplesPerSet)(1)

  //Get the trainingData in the right format for the spam mails
  var spamData = spamMails.map(x => bag.feature(x._2.split(" "))).toArray

  //Get the trainingData in the right format for the ham mails
  var hamData = hamMails.map(x => bag.feature(x._2.split(" "))).toArray

  //Combine the training data from both categories
  var trainingData = spamData ++ hamData

//Create the bayes model as a multinomial with 2 classification groups and the amount of features passed in the constructor.
  var bayes = new NaiveBayes(NaiveBayes.Model.MULTINOMIAL, 2, data.size)
  //Now train the bayes instance with the training data, which is represented in a specific format due to the bag.feature method, and the known classifiers.
  bayes.learn(trainingData, classifiers)



  //Now we are ready for evaluation, for this we will use the testing sets:


  val listOfSpam2Files =   getFilesFromDir(easyHam2Path)
  //Then get the messages that are contained in these files
  val spam2Mails = listOfSpam2Files.map{x => (x,getMessage(x)) }

  val spam2FeatureVectors = spam2Mails.map(x => bag.feature(x._2.split(" ")))

  val spam2ClassificationResults = spam2FeatureVectors.map(x => bayes.predict(x))

  val spamClassifications = spam2ClassificationResults.count( x=> x == 0);
  println(spamClassifications + " of " + listOfSpam2Files.length + "were classified as spam")
  println(((spamClassifications.toDouble /  listOfSpam2Files.length) * 100)  + "% was classified as spam")

  val hamClassifications = spam2ClassificationResults.count( x=> x == 1);
  println(hamClassifications + " of " + listOfSpam2Files.length + "were classified as ham")
  println(((hamClassifications.toDouble /  listOfSpam2Files.length) * 100)  + "% was classified as ham")

  val unknownClassifications = spam2ClassificationResults.count( x=> x == -1);
  println(unknownClassifications + " of " + listOfSpam2Files.length + "were unknownly classified")
  println(((unknownClassifications.toDouble /  listOfSpam2Files.length) * 100)  + "% was unknownly classified")

  val testmessage =  bag.feature("this email smells like i want to offer you something for money You want to buy some viagra from me?".split(" "))
  val result = bayes.predict(testmessage)
  if (result == 1) {
    println("HAM")
  }
  else if (result == 0)
  {
    println("SPAM")
  }
  else
  {
    println("unknown: " + result)
  }

  }
*/
  def getFilesFromDir(path: String):List[File] = {
    val d = new File(path)
    if (d.exists && d.isDirectory) {
      //Remove the mac os basic storage file, and alternatively for unix systems "cmds"
      d.listFiles.filter(_.isFile).toList.filter(x => ! x.toString.contains(".DS_Store") && ! x.toString.contains("cmds"))
    } else {
      List[File]()
    }
  }

  def getStopWords() : List[String] =
  {
    val source = scala.io.Source.fromFile(new File("/Users/mikedewaard/MachineLearning/Example Data/stopwords.txt"))("latin1")
    val lines = source.mkString.split("\n")
    source.close()
    return  lines.toList
  }

  def getMessage(file : File)  : String  =
  {
    //Note that the encoding of the example files is latin1, thus this should be passed to the from file method.
    val source = scala.io.Source.fromFile(file)("latin1")
    val lines = source.getLines mkString "\n"
    source.close()
    //Find the first line break in the email, as this indicates the message body
    val firstLineBreak = lines.indexOf("\n\n")
    //Return the message body filtered by only text from a-z and to lower case
    return lines.substring(firstLineBreak).replace("\n"," ").replaceAll("[^a-zA-Z ]","").toLowerCase()
  }
}
