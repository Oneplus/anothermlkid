## NN4NER

A NER using netural network. This project follows the course project in [CS224N: Natural Language Processing](http://class.stanford.edu/cs224n/Fall2012) - pa4

I rearranged the project structure slightly.

In following parameter configuration:

* Window Size : 5
* Hidden Layer Nodes : 100
* Regularization : 0.01
* Iterations : 10
* Learning Rate : 0.001

this model achieve an F1-score of 0.7164

## Complie And Run

$ ant
$ java -cp classes:extlib/ejml.jar NER ./data/train ./data/dev [regular] [iteration] [rate]

## Original Instruction (README.txt)

We've set things up so that it just works out of the box with Eclipse and Ant. 


ECLIPSE:
Go to File->New->New Project (Not Java Project). Select Java Project from existing Ant Buildfile and then select the build.xml in this folder. 
To set up a run configuration for NER, right-click NER.java in the project explorer and select "Run As" -> "Run Configuration". 
Under the Main tab, type "NER" for project and "NER" for main class. Then, proceed to the arguments tab and type "../data/train ../data/dev -print" for program arguments and "-Xmx1G" for VM arguments without the quotations. 
Then, click "Apply" and "Run" buttons. This will run your program.
Once you have run configuration you 
 
ANT:
we've provided a basic build.xml file for use with ant.  Just call:

$ ant

COMMAND LINE:
If you want to develop on the command line, use the following commands:
$ mkdir classes
$ ant
$ java -cp classes -Xmx1G NER ../data/train ../data/dev -print
