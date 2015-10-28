import java.awt.{Rectangle}
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import smile.plot.BarPlot

import scala.swing.{MainFrame, SimpleSwingApplication}
import scala.util.Try

object RecommendationSystem extends SimpleSwingApplication {


  case class EmailData(emailDate : Date, sender : String, subject : String, body : String)

  def top = new MainFrame {
    title = "Recommendation System Example from http://xyclade.ml"

    val basePath = "data"
    val easyHamPath = basePath + "/easy_ham"

    try
      {
    val mails = getFilesFromDir(easyHamPath).map(x => getFullEmail(x))
    val timeSortedMails = mails
      .map(x => EmailData(getDateFromEmail(x), getSenderFromEmail(x), getSubjectFromEmail(x), getMessageBodyFromEmail(x)))
      .sortBy(x => x.emailDate)

    val (trainingData, testingData) = timeSortedMails
      .splitAt(timeSortedMails.length / 2)


    //First we group the emails by Sender, then we extract only the sender address and amount of emails, and finally we sort them on amounts ascending
    val mailsGroupedBySender = trainingData
      .groupBy(x => x.sender)
      .map(x => (x._1, Math.log1p(x._2.length)))
      .toArray
      .sortBy(x => x._2)

    //In order to plot the data we split the values from the addresses as this is how the plotting library accepts the data.
    val senderDescriptions = mailsGroupedBySender.map(x => x._1)
    val senderValues = mailsGroupedBySender.map(x => x._2.toDouble)

    val barPlot = BarPlot.plot("", senderValues, senderDescriptions)

    //Rotate the email addresses by -80 degrees such that we can read them
    barPlot.getAxis(0).setRotation(-1.3962634)
    barPlot.setAxisLabel(0, "")
    barPlot.setAxisLabel(1, "Amount of emails received on log Scale ")
    peer.setContentPane(barPlot)

    bounds = new Rectangle(800, 600)

    val mailsGroupedByThread = trainingData
      .groupBy(x => x.subject)

    //Create a list of tuples with (subject, list of emails)
    val threadBarPlotData = mailsGroupedByThread
      .map(x => (x._1, Math.log1p(x._2.length)))
      .toArray
      .sortBy(x => x._2)

    val threadDescriptions = threadBarPlotData
      .map(x => x._1)
    val threadValues = threadBarPlotData
      .map(x => x._2.toDouble)

    val mailGroupsWithMinMaxDates = mailsGroupedByThread
      .map(x => (x._1, x._2, (x._2
      .maxBy(x => x.emailDate)
      .emailDate.getTime - x._2.minBy(x => x.emailDate).emailDate.getTime
      ) / 1000))

    //turn into a list of tuples with (topic, list of emails, time difference, and weight) filtered that only threads occur
    val threadGroupsWithWeights = mailGroupsWithMinMaxDates
      .filter(x => x._3 != 0)
      .map(x => (x._1, x._2, x._3, 10 + Math.log10(x._2.length.toDouble / x._3)))


    val stopWords = getStopWords

    val threadTermWeights = threadGroupsWithWeights
      .toArray
      .sortBy(x => x._4)
      .flatMap(x => x._1
      .replaceAll("[^a-zA-Z ]", "")
      .toLowerCase.split(" ")
      .filter(_.nonEmpty)
      .map(y => (y, x._4)))

    val filteredThreadTermWeights = threadTermWeights
      .groupBy(x => x._1)
      .map(x => (x._1, x._2.maxBy(y => y._2)._2))
      .toArray.sortBy(x => x._1)
      .filter(x => !stopWords.contains(x._1))


    val tdm = trainingData
      .flatMap(x => x.body.split(" "))
      .filter(x => x.nonEmpty && !stopWords.contains(x))
      .groupBy(x => x)
      .map(x => (x._1, Math.log10(x._2.length + 1)))
      .filter(x => x._2 != 0)


    val trainingRanks = trainingData.map(mail => {
      //mail contains (full content, date, sender, subject, body)

      //Determine the weight of the sender
      val senderWeight = mailsGroupedBySender
        .collectFirst { case (mail.sender, x) => x + 1}
        .getOrElse(1.0)

      //Determine the weight of the subject
      val termsInSubject = mail.subject
        .replaceAll("[^a-zA-Z ]", "")
        .toLowerCase.split(" ")
        .filter(x => x.nonEmpty && !stopWords.contains(x))

      val termWeight = if (termsInSubject.size > 0) termsInSubject
        .map(x => {
        tdm.collectFirst { case (y, z) if y == x => z + 1}
          .getOrElse(1.0)
      })
        .sum / termsInSubject.length
      else 1.0

      //Determine if the email is from a thread, and if it is the weight from this thread:
      val threadGroupWeight: Double = threadGroupsWithWeights
        .collectFirst { case (mail.subject, _, _, weight) => weight}
        .getOrElse(1.0)

      //Determine the commonly used terms in the email and the weight belonging to it:
      val termsInMailBody = mail.body
        .replaceAll("[^a-zA-Z ]", "")
        .toLowerCase.split(" ")
        .filter(x => x.nonEmpty && !stopWords.contains(x))

      val commonTermsWeight = if (termsInMailBody.size > 0) termsInMailBody
        .map(x => {
        tdm.collectFirst { case (y, z) if y == x => z + 1}
          .getOrElse(1.0)
      })
        .sum / termsInMailBody.length
      else 1.0

      val rank = termWeight * threadGroupWeight * commonTermsWeight * senderWeight

      (mail, rank)
    })

    val sortedTrainingRanks = trainingRanks.sortBy(x => x._2)

    val median = sortedTrainingRanks(sortedTrainingRanks.length / 2)._2
    val mean = sortedTrainingRanks.map(x => x._2).sum / sortedTrainingRanks.length
    println("Median:" + median + "  Mean:" + mean)


    val testingRanks = testingData.map(mail => {
      //mail contains (full content, date, sender, subject, body)

      //Determine the weight of the sender
      val senderWeight = mailsGroupedBySender
        .collectFirst { case (mail.sender, x) => x +1}
        .getOrElse(1.0)

      //Determine the weight of the subject
      val termsInSubject = mail.subject
        .replaceAll("[^a-zA-Z ]", "")
        .toLowerCase.split(" ")
        .filter(x => x.nonEmpty && !stopWords.contains(x))

      val termWeight = if (termsInSubject.size > 0) termsInSubject
        .map(x => {
        tdm.collectFirst { case (y, z) if y == x => z + 1}
          .getOrElse(1.0)
      })
        .sum / termsInSubject.length
      else 1.0

      //Determine if the email is from a thread, and if it is the weight from this thread:
      val threadGroupWeight: Double = threadGroupsWithWeights
        .collectFirst { case (mail.subject, _, _, weight) => weight}
        .getOrElse(1.0)

      //Determine the commonly used terms in the email and the weight belonging to it:
      val termsInMailBody = mail.body
        .replaceAll("[^a-zA-Z ]", "")
        .toLowerCase.split(" ")
        .filter(x => x.nonEmpty && !stopWords.contains(x))

      val commonTermsWeight = if (termsInMailBody.size > 0) termsInMailBody
        .map(x => {
        tdm.collectFirst { case (y, z) if y == x => z + 1}
          .getOrElse(1.0)
      })
        .sum / termsInMailBody.length
      else 1.0

      val rank = termWeight * threadGroupWeight * commonTermsWeight * senderWeight

      (mail, rank, termWeight,threadGroupWeight,commonTermsWeight,senderWeight)
    })

    val priorityEmails = testingRanks.filter(x => x._2 >= mean/2).sortBy(x => -x._2)
    val df = new java.text.DecimalFormat("#.##")

    println("|Date | Sender  | Subject  | Rank | thread term  | thread time  | common terms  | sender |")
    println("| :--- | : -- | :--  | :-- | :-- |  :-- |  :-- |  :-- |  ")
    priorityEmails.take(10).foreach(x => println("| " + x._1.emailDate + " | " + x._1.sender + " | " + x._1.subject + " | " + df.format(x._2) + " |"+ df.format(x._3) + " |"+ df.format(x._4) + " |"+ df.format(x._5) + " |"+ df.format(x._6) + " |"))


    println(priorityEmails.length + " ranked as priority")

  }
    catch
      {
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
    val subject = email
      .substring(subjectIndex + 8, endOfSubjectIndex)
      .trim
      .toLowerCase

    //Additionally, we check whether the email was a response and remove the 're: ' tag, to make grouping on topic easier:
    subject.replace("re: ", "")
  }

  def getMessageBodyFromEmail(email: String): String = {

    val firstLineBreak = email.indexOf("\n\n")
    //Return the message body filtered by only text from a-z and to lower case
    email.substring(firstLineBreak)
      .replace("\n", " ")
      .replaceAll("[^a-zA-Z ]", "")
      .toLowerCase
  }


  def getSenderFromEmail(email: String): String = {
    //Find the index of the From: line
    val fromLineIndex = email.indexOf("From:")
    val endOfLine = email.substring(fromLineIndex).indexOf('\n') + fromLineIndex

    //Search for the <> tags in this line, as if they are there, the email address is contained inside these tags
    val mailAddressStartIndex = email
      .substring(fromLineIndex, endOfLine)
      .indexOf('<') + fromLineIndex + 1
    val mailAddressEndIndex = email
      .substring(fromLineIndex, endOfLine)
      .indexOf('>') + fromLineIndex

    if (mailAddressStartIndex > mailAddressEndIndex) {

      //The email address was not embedded in <> tags, extract the substring without extra spacing and to lower case
      var emailString = email
        .substring(fromLineIndex + 5, endOfLine)
        .trim
        .toLowerCase

      //Remove a possible name embedded in () at the end of the line, for example in test@test.com (tester) the name would be removed here
      val additionalNameStartIndex = emailString.indexOf('(')
      if (additionalNameStartIndex == -1) {
        emailString
          .toLowerCase
      }
      else {
        emailString
          .substring(0, additionalNameStartIndex)
          .trim
          .toLowerCase
      }
    }
    else {
      //Extract the email address from the tags. If these <> tags are there, there is no () with a name in the From: string in our data
      email
        .substring(mailAddressStartIndex, mailAddressEndIndex)
        .trim
        .toLowerCase
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

  def getStopWords: List[String] = {
    val source = scala.io.Source.fromFile(new File("data/stopwords.txt"))("latin1")
    val lines = source.mkString.split("\n")
    source.close()
    lines.toList
  }
}

