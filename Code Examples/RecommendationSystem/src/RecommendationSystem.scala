import java.awt.{Rectangle, ComponentOrientation}
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import smile.plot.BarPlot

import scala.swing.{MainFrame, SimpleSwingApplication}
import scala.util.Try

object RecommendationSystem extends SimpleSwingApplication {

  def top = new MainFrame {
    title = "Recommendation System Example"

    val basePath = "/Users/mikedewaard/ML_for_Hackers/03-Classification/data"
    val easyHamPath = basePath + "/easy_ham"
    val easyHam2Path = basePath + "/easy_ham_2"
    val hardHamPath = basePath + "/hard_ham"
    val hardHam2Path = basePath + "/hard_ham_2"


    val listOfSpamFiles = getFilesFromDir(easyHamPath)

    val mailBodies = listOfSpamFiles.map(x => getFullEmail(x))

    val mailInformation = mailBodies.map(x => (x, getDateFromEmail(x), getSenderFromEmail(x), getSubjectFromEmail(x), getMessageBodyFromEmail(x)))

    val timeSortedMails = mailInformation.sortBy(x => x._2)

    val testAndTrainingSplit = timeSortedMails.splitAt(timeSortedMails.length / 2)


    val trainingData = testAndTrainingSplit._1

    val testingData = testAndTrainingSplit._2

    val mailsGroupedBySender = trainingData.groupBy(x => x._3).map(x => (x._1, Math.log1p(x._2.length))).toArray.sortBy(x => x._2)
    val senderDescriptions = mailsGroupedBySender.map(x => x._1)
    val senderValues = mailsGroupedBySender.map(x => x._2.toDouble)

    val mailsGroupedByThread = trainingData.groupBy(x => x._4)

    //Create a list of tuples with (subject, list of emails, time difference between first and last email)
    val mailGroupsWithMinMaxDates = mailsGroupedByThread.map(x => (x._1, x._2, (x._2.maxBy(x => x._2)._2.getTime - x._2.minBy(x => x._2)._2.getTime) / 1000))

    //turn into a list of tuples with (topic, list of emails, time difference, and weight) filtered that only threads occur
    val threadGroupsWithWeights = mailGroupsWithMinMaxDates.filter(x => x._3 != 0).map(x => (x._1, x._2, x._3, 10 + Math.log10(x._2.length.toDouble / x._3)))


    val StopWords = getStopWords
    val termWeights = threadGroupsWithWeights.toArray.sortBy(x => x._4).flatMap(x => x._1.replaceAll("[^a-zA-Z ]", "").toLowerCase.split(" ").filter(_.nonEmpty).map(y => (y, x._4)))
    val filteredTermWeights = termWeights.groupBy(x => x._1).map(x => (x._1, x._2.maxBy(y => y._2)._2)).toArray.sortBy(x => x._1).filter(x => !StopWords.contains(x._1))


    val mailTDM = new TDM()
    //Build up the Term-Document Matrix for the training emails
    trainingData.foreach(x => x._5.split(" ").filter(_.nonEmpty).foreach(y => mailTDM.addTermToRecord(y)))
    //Filter out all stop words
    mailTDM.records = mailTDM.records.filter(x => !StopWords.contains(x.term))
    //Filter out the stopwords and get their frequency with a log filter, such that low occurrence words are removed.
    val filteredCommonTerms = mailTDM.records.map(x => (x.term, x.log10Frequency)).filter(x => x._2 != 0).sortBy(x => x._1)
    //  filteredCommonTerms.foreach(x => println(x._1 + ", " + x._2))

    val alsoRecord = mailTDM.records.filter(x => x.term == "also")(0)
    val totalFrequencyForAlso = alsoRecord.frequencyInAllDocuments
    val logFrequency = alsoRecord.log10Frequency


    println(logFrequency)

    val threadBarPlotData = mailsGroupedByThread.map(x => (x._1, x._2.length)).toArray.sortBy(x => x._2)
    val threadDescriptions = threadBarPlotData.map(x => x._1)
    val threadValues = threadBarPlotData.map(x => Math.log1p(x._2.toDouble))

    val weightedThreadBarPlotData = threadGroupsWithWeights.toArray.sortBy(x => x._4)
    val weightedThreadDescriptions = weightedThreadBarPlotData.map(x => x._1)
    val weightedThreadValues = weightedThreadBarPlotData.map(x => x._4)


    val barPlot = BarPlot.plot("Amount of emails per subject on log scale", weightedThreadValues, weightedThreadDescriptions)
    //Rotate the email addresses by -80 degrees such that we can read them
    barPlot.getAxis(0).setRotation(-1.3962634)
    barPlot.setAxisLabel(0, "")
    barPlot.setAxisLabel(1, "Weighted amount of mails per subject ")
    peer.setContentPane(barPlot)

    bounds = new Rectangle(800, 600)


    //Given all feature data we have, it's time to clean up and merge the data to calculate
    //mailsGroupedBySender =  list(sender address, log of amount of emails sent)
    //threadGroupsWithWeights = list(Thread name, list of emails, time difference, weight)
    //filteredTermWeights = list(term, weight) for subject
    //filteredCommonTerms = list(term,weight) for email body
    val combinedFeatures = trainingData.map(mail => {
      //mail contains (full content, date, sender, subject, body)

      var termWeight = 1.0
      var threadGroupweight = 1.0
      var commonTermsWeight = 1.0
      var senderWeight = 1.0

      //Determine the weight of the sender
      val calculatedSenderWeight = mailsGroupedBySender.collectFirst { case (mail._3, x) => x}
      if (calculatedSenderWeight.nonEmpty) {
        senderWeight = calculatedSenderWeight.get
      }

      //Determine the weight of the subject
      val termsInSubject = mail._4.replaceAll("[^a-zA-Z ]", "").toLowerCase.split(" ").filter(_.nonEmpty).filter(x => !StopWords.contains(x))
      val calculatedTermWeight = termsInSubject.map(x => {
        val weight = filteredTermWeights.collectFirst { case (y, z) if (y == x) => z}
        if (weight.nonEmpty) {
          weight.get
        }
        else {
          0
        }
      }).sum / termsInSubject.length

      if (calculatedTermWeight > 0) {
        termWeight = calculatedTermWeight
      }

      //Determine if the email is from a thread, and if it is the weight from this thread:
      val threadWeight = threadGroupsWithWeights.collectFirst { case (threadName, _, _, weight) if threadName == mail._4 => weight}
      if (threadWeight.nonEmpty) {
        threadGroupweight =  threadWeight.get
      }


      //Determine the commonly used terms in the email and the weight belonging to it:
      val termsInMailBody = mail._5.replaceAll("[^a-zA-Z ]", "").toLowerCase.split(" ").filter(_.nonEmpty).filter(x => !StopWords.contains(x))
      val calculatedCommonTermWeight = termsInMailBody.map(x => {
        val weight = filteredCommonTerms.collectFirst { case (y, z) if (y == x) => z}
        if (weight.nonEmpty) {
          weight.get
        }
        else {
          0
        }
      }).sum / termsInMailBody.length
      if (calculatedCommonTermWeight > 0) {

        commonTermsWeight = calculatedCommonTermWeight
      }
      (mail, termWeight, threadGroupweight, commonTermsWeight, senderWeight)
    })


    //combinedFeatures.foreach(x => println("termWeight: " + x._2 + "\t threadGroupweight: " + x._3 + "\t commontermsWeight:  " + x._4 + "\t senderWeight: " + x._5 + "\t " + x._1._4))
    val mailRanks = combinedFeatures.map(x => (x._1._4, x._2 * x._3 * x._4 * x._5 ))
    mailRanks.sortBy(x => x._2 ).foreach(x => println(x._2 + "\t\t\t" + x._1))
  }


  def getStopWords: List[String] = {
    val source = scala.io.Source.fromFile(new File("/Users/mikedewaard/MachineLearning/Example Data/stopwords.txt"))("latin1")
    val lines = source.mkString.split("\n")
    source.close()
    lines.toList
  }


  def getFilesFromDir(path: String): List[File] = {
    val d = new File(path)
    if (d.exists && d.isDirectory) {
      //Remove the mac os basic storage file, and alternatively for unix systems "cmds"
      d.listFiles.filter(_.isFile).toList.filter(x => !x.toString.contains(".DS_Store") && !x.toString.contains("cmds"))
    } else {
      List[File]()
    }
  }


  def getFullEmail(file: File): String = {
    //Note that the encoding of the example files is latin1, thus this should be passed to the from file method.
    val source = scala.io.Source.fromFile(file)("latin1")
    val fullEmail = source.getLines mkString "\n"
    source.close()

    fullEmail
  }


  def getSubjectFromEmail(email: String): String = {

    //Find the index of the end of the subject line
    val subjectIndex = email.indexOf("Subject:")
    val endOfSubjectIndex = email.substring(subjectIndex).indexOf('\n') + subjectIndex

    //Extract the subject: start of subject + 7 (length of Subject:) until the end of the line.
    val subject = email.substring(subjectIndex + 8, endOfSubjectIndex).trim.toLowerCase

    //Additionally, we check whether the email was a response and remove the 're: ' tag, to make grouping on topic easier:
    subject.replace("re: ", "")
  }

  def getMessageBodyFromEmail(email: String): String = {

    val firstLineBreak = email.indexOf("\n\n")
    //Return the message body filtered by only text from a-z and to lower case
    email.substring(firstLineBreak).replace("\n", " ").replaceAll("[^a-zA-Z ]", "").toLowerCase
  }


  def getSenderFromEmail(email: String): String = {
    //Find the index of the From: line
    val fromLineIndex = email.indexOf("From:")
    val endOfLine = email.substring(fromLineIndex).indexOf('\n') + fromLineIndex

    //Search for the <> tags in this line, as if they are there, the email address is contained inside these tags
    val mailAddressStartIndex = email.substring(fromLineIndex, endOfLine).indexOf('<') + fromLineIndex + 1
    val mailAddressEndIndex = email.substring(fromLineIndex, endOfLine).indexOf('>') + fromLineIndex

    if (mailAddressStartIndex > mailAddressEndIndex) {

      //The email address was not embedded in <> tags, extract the substring without extra spacing and to lower case
      var emailString = email.substring(fromLineIndex + 5, endOfLine).trim.toLowerCase

      //Remove a possible name embedded in () at the end of the line, for example in test@test.com (tester) the name would be removed here
      val additionalNameStartIndex = emailString.indexOf('(')
      if (additionalNameStartIndex == -1) {
        emailString.toLowerCase
      }
      else {
        emailString.substring(0, additionalNameStartIndex).trim.toLowerCase
      }
    }
    else {
      //Extract the email address from the tags. If these <> tags are there, there is no () with a name in the From: string in our data
      email.substring(mailAddressStartIndex, mailAddressEndIndex).trim.toLowerCase
    }
  }

  def getDateFromEmail(email: String): Date = {
    //Find the index of the Date: line in the complete email
    val dateLineIndex = email.indexOf("Date:")
    val endOfDateLine = email.substring(dateLineIndex).indexOf('\n') + dateLineIndex

    //All possible date patterns in the emails.
    val datePatterns = Array("EEE MMM dd HH:mm:ss yyyy", "EEE, dd MMM yyyy HH:mm", "dd MMM yyyy HH:mm:ss", "EEE MMM dd yyyy HH:mm")

    datePatterns.foreach { x =>
      //Try to directly return a date from the formatting.when it fails on a pattern it continues with the next one until one works
      Try(return new SimpleDateFormat(x).parse(email.substring(dateLineIndex + 5, endOfDateLine).trim.substring(0, x.length)))
    }
    //Finally, if all failed return null (this will not happen with our example data but without this return the code will not compile)
    null
  }
}

