import java.io.File
import smile.classification.NaiveBayes
import smile.feature.Bag

object NaiveBayesExample {


  def main(args: Array[String]): Unit = {
    val basePath = "data/"
    val spamPath = basePath + "/spam"
    val easyHamPath = basePath + "/easy_ham"
    val easyHam2Path = basePath + "/easy_ham_2"

    val amountOfSamplesPerSet = 500
    val amountOfFeaturesToTake = 400

    try {
      //First get a subset of the file names for the spam sample set (500 is the complete set in this case)
      val listOfSpamFiles = getFilesFromDir(spamPath).take(amountOfSamplesPerSet)
      //Then get the messages that are contained in these files
      val spamMails = listOfSpamFiles.map(x => (x, getMessage(x)))

      val stopWords = getStopWords
      val spamTDM = spamMails
        .flatMap(email => email
          ._2.split(" ")
          .filter(word => word.nonEmpty && !stopWords.contains(word))
          .map(word => (email._1.getName, word)))
        .groupBy(x => x._2)
        .map(x => (x._1, x._2.groupBy(x => x._1)))
        .map(x => (x._1, x._2.map(y => (y._1, y._2.length)))).toList

      //Sort the words by occurrence rate  descending (amount of times the word occurs among all documents)
      val sortedSpamTDM = spamTDM.sortBy(x => -(x._2.size.toDouble / spamMails.length))
      val spamFeatures = sortedSpamTDM.take(amountOfFeaturesToTake).map(x => x._1)

      //Get a subset of the file names from the ham sample set (note that in this case it is not necessary to randomly sample as the emails are already randomly ordered)
      val listOfHamFiles = getFilesFromDir(easyHamPath).take(amountOfSamplesPerSet)

      //Get the messages that are contained in the ham files
      val hamMails = listOfHamFiles.map(x => (x, getMessage(x)))
      //Then its time for feature selection specifically for the Ham messages
      val hamTDM = hamMails
        .flatMap(email => email
          ._2.split(" ")
          .filter(word => word.nonEmpty && !stopWords.contains(word))
          .map(word => (email._1.getName, word)))
        .groupBy(x => x._2)
        .map(x => (x._1, x._2.groupBy(x => x._1)))
        .map(x => (x._1, x._2.map(y => (y._1, y._2.length)))).toList

      //Sort the words by occurrence rate  descending (amount of times the word occurs among all documents)
      val sortedHamTDM = hamTDM.sortBy(x => -(x._2.size.toDouble / spamMails.length))
      val hamFeatures = sortedHamTDM.take(amountOfFeaturesToTake).map(x => x._1)

      //Now we have a set of ham and spam features, we group them and then remove the intersecting features, as these are noise.
      var data = (hamFeatures ++ spamFeatures).toSet
      hamFeatures.intersect(spamFeatures).foreach(x => data = data - x)


      //Initialize a bag of words that takes the top x features from both spam and ham and combines them
      val bag = new Bag[String](data.toArray)

      //Initialize the classifier array with first a set of 0(spam) and then a set of 1(ham) values that represent the emails
      val classifiers = Array.fill[Int](amountOfSamplesPerSet)(0) ++ Array.fill[Int](amountOfSamplesPerSet)(1)

      //Get the trainingData in the right format for the spam mails
      val spamData = spamMails.map(x => bag.feature(x._2.split(" "))).toArray

      //Get the trainingData in the right format for the ham mails
      val hamData = hamMails.map(x => bag.feature(x._2.split(" "))).toArray

      //Combine the training data from both categories
      val trainingData = spamData ++ hamData

      //Create the bayes model as a multinomial with 2 classification groups and the amount of features passed in the constructor.
      val bayes = new NaiveBayes(NaiveBayes.Model.MULTINOMIAL, 2, data.size)
      //Now train the bayes instance with the training data, which is represented in a specific format due to the bag.feature method, and the known classifiers.
      bayes.learn(trainingData, classifiers)



      //Now we are ready for evaluation, for this we will use the testing sets:
      val listOfSpam2Files = getFilesFromDir(easyHam2Path)
      //Then get the messages that are contained in these files
      val spam2Mails = listOfSpam2Files.map { x => (x, getMessage(x)) }

      val spam2FeatureVectors = spam2Mails.map(x => bag.feature(x._2.split(" ")))

      val spam2ClassificationResults = spam2FeatureVectors.map(x => bayes.predict(x))

      val spamClassifications = spam2ClassificationResults.count(x => x == 0)
      println(spamClassifications + " of " + listOfSpam2Files.length + " were classified as spam")
      println(((spamClassifications.toDouble / listOfSpam2Files.length) * 100) + "% was classified as spam")

      val hamClassifications = spam2ClassificationResults.count(x => x == 1)
      println(hamClassifications + " of " + listOfSpam2Files.length + " were classified as ham")
      println(((hamClassifications.toDouble / listOfSpam2Files.length) * 100) + "% was classified as ham")

      val unknownClassifications = spam2ClassificationResults.count(x => x == -1)
      println(unknownClassifications + " of " + listOfSpam2Files.length + " were unknowingly classified")
      println(((unknownClassifications.toDouble / listOfSpam2Files.length) * 100) + "% was unknowingly classified")
    }
    catch {
      case e: Exception => println("You probably are missing the sample data. You can download these from the spamassasin corpus (mentioned in the example on http://xyclade.ml) and place them in the directory 'data' in this project. Check the exception for more details: " + e);
    }

  }

  def getFilesFromDir(path: String): List[File] = {
    val d = new File(path)
    if (d.exists && d.isDirectory) {
      //Remove the mac os basic storage file, and alternatively for unix systems "cmds"
      d.listFiles.filter(x => x.isFile && !x.toString.contains(".DS_Store") && !x.toString.contains("cmds")).toList
    } else {
      List[File]()
    }
  }

  def getStopWords: List[String] = {
    val source = scala.io.Source.fromFile(new File("data/stopwords.txt"))("latin1")
    val lines = source.mkString.split("\n")
    source.close()
    lines.toList
  }

  def getMessage(file: File): String = {
    //Note that the encoding of the example files is latin1, thus this should be passed to the from file method.
    val source = scala.io.Source.fromFile(file)("latin1")
    val lines = source.getLines mkString "\n"
    source.close()
    //Find the first line break in the email, as this indicates the message body
    val firstLineBreak = lines.indexOf("\n\n")
    //Return the message body filtered by only text from a-z and to lower case
    lines.substring(firstLineBreak).replace("\n", " ").replaceAll("[^a-zA-Z ]", "").toLowerCase
  }
}
