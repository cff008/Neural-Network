/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */

import java.util.*;


public class NNImpl{
	public ArrayList<Node> inputNodes=null;//list of the output layer nodes.
	public ArrayList<Node> hiddenNodes=null;//list of the hidden layer nodes
	public ArrayList<Node> outputNodes=null;// list of the output layer nodes

	public ArrayList<Instance> trainingSet=null;//the training set

	Double learningRate=1.0; // variable to store the learning rate
	int maxEpoch=1; // variable to store the maximum number of epochs

	/**
	 * This constructor creates the nodes necessary for the neural network
	 * Also connects the nodes of different layers
	 * After calling the constructor the last node of both inputNodes and  
	 * hiddenNodes will be bias nodes. 
	 */

	public NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[][] outputWeights)
	{
		this.trainingSet=trainingSet;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;

		//input layer nodes
		inputNodes=new ArrayList<Node>();
		int inputNodeCount=trainingSet.get(0).attributes.size();
		int outputNodeCount=trainingSet.get(0).classValues.size();
		for(int i=0;i<inputNodeCount;i++)
		{
			Node node=new Node(0);
			inputNodes.add(node);
		}

		//bias node from input layer to hidden
		Node biasToHidden=new Node(1);
		inputNodes.add(biasToHidden);

		//hidden layer nodes
		hiddenNodes=new ArrayList<Node> ();
		for(int i=0;i<hiddenNodeCount;i++)
		{
			Node node=new Node(2);
			//Connecting hidden layer nodes with input layer nodes
			for(int j=0;j<inputNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(inputNodes.get(j),hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}

		//bias node from hidden layer to output
		Node biasToOutput=new Node(3);
		hiddenNodes.add(biasToOutput);

		//Output node layer
		outputNodes=new ArrayList<Node> ();
		for(int i=0;i<outputNodeCount;i++)
		{
			Node node=new Node(4);
			//Connecting output layer nodes with hidden layer nodes
			for(int j=0;j<hiddenNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}	
			outputNodes.add(node);
		}	
	}
	/**
	 * Get the output from the neural network for a single instance
	 * Return the idx with highest output values. For example if the outputs
	 * of the outputNodes are [0.1, 0.5, 0.2], it should return 1. If outputs
	 * of the outputNodes are [0.1, 0.5, 0.5], it should return 2. 
	 * The parameter is a single instance. 
	 */

	public int calculateOutputForInstance(Instance inst)
	{
		//Get input values and put them into the inputNodes array.
		for(int i = 0; i < inst.attributes.size(); i++){
			inputNodes.get(i).setInput(inst.attributes.get(i));
		}

		for(int i = 0; i < hiddenNodes.size(); i++){
			hiddenNodes.get(i).calculateOutput();
		}

		for(int i = 0; i < outputNodes.size();i++){
			outputNodes.get(i).calculateOutput();
		}

		int maxInt = 0;
		double maxVal = 0.0;
		for(int i = 0; i < outputNodes.size();i++){
			outputNodes.get(i).calculateOutput();
			if(outputNodes.get(i).getOutput() > maxVal){
				maxVal = outputNodes.get(i).getOutput();
				maxInt = i;
			}
		}
		return maxInt;	
	}
	/**
	 * Train the neural networks with the given parameters
	 * 
	 * The parameters are stored as attributes of this class
	 */	
	public void train()
	{

		for(int i = 0; i < maxEpoch; i ++){
			for(Instance temp : trainingSet){

				for (int j = 0; j < inputNodes.size() - 1; j++) { 
					inputNodes.get(j).setInput(temp.attributes.get(j));
				}



				for (int j = 0; j < hiddenNodes.size() - 1; j++) {
					hiddenNodes.get(j).calculateOutput();
				}
				List<Double> dJ = new ArrayList<>();
				dJ = deltaJ(temp);

				List<Double> dI = new ArrayList<>();
				dI = deltaI(temp,dJ);

				reWeight(dI);
				reWeight2(dJ);				
			}
		}
	}

	private void reWeight(List<Double> temp){
		for (int t = 0; t < hiddenNodes.size() - 1; t++) {
			for (NodeWeightPair curr: hiddenNodes.get(t).parents) {
				curr.weight += learningRate * curr.node.getOutput() * temp.get(t);
			}
		}
	}

	private void reWeight2(List<Double> temp){
		for (int t = 0; t < outputNodes.size() - 1; t++) {
			for (NodeWeightPair curr: outputNodes.get(t).parents) {
				curr.weight += learningRate * curr.node.getOutput() * temp.get(t);
			}
		}
	}

	public List<Double> deltaJ(Instance temp){

		List<Double> dList = new ArrayList<>();
		for(int i = 0; i < outputNodes.size(); i++){
			double dj = 0;
			outputNodes.get(i).calculateOutput();
			dj = relu(outputNodes.get(i).getSum())*(temp.classValues.get(i) - outputNodes.get(i).getOutput());
			dList.add(dj);
		}
		return dList;
	}

	public List<Double> deltaI(Instance temp, List<Double> deltaJ){
		List<Double> dList = new ArrayList<>();
		for (int i = 0; i < hiddenNodes.size(); i++) {	
			double di = 0;
			double val = 0;
			for (int j = 0; j < outputNodes.size(); j++) {
				val += outputNodes.get(j).parents.get(i).weight * deltaJ.get(j);
			}
			di = relu(hiddenNodes.get(i).getSum()) *val;
			dList.add(di);
		}
		return dList;
	}

	public double relu(Double temp){
		if(temp < 0){
			return 0.0;
		}
		return temp;
	}	
}