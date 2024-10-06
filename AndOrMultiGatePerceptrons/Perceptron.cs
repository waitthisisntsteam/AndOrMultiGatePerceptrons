using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AndOrMultiGatePerceptrons
{
    class Perceptron
    {
        double[] Weights;
        double Bias;
        double MutationAmount;
        Func<double, double, double> ErrorFunc;

        public Perceptron(double[] weights, double bias, double mutationAmount, Func<double, double, double> errorFunc)
        {
            Weights = weights;
            Bias = bias;
            MutationAmount = mutationAmount;
            ErrorFunc = errorFunc;
        }
        public Perceptron(int inputAmount, double mutationAmount, Func<double, double, double> errorFunc)
        {
            Weights = new double[inputAmount];
            Bias = 0;

            MutationAmount = mutationAmount;
            ErrorFunc = errorFunc;
        }

        private double Random(Random random, double min, double max) => (random.NextDouble() * (max - min)) + min;
        public void Randomize(Random random, double min, double max)
        {
            for (int i = 0; i < Weights.Length; i++) Weights[i] = Random(random, min, max);
            Bias = Random(random, min, max);
        }

        public double[] UserCompute(double[][] inputs)
        {
            double[] outputs = Compute(inputs);

            for (int i = 0; i < outputs.Length; i++)
            {
                if (outputs[i] < 0.5) outputs[i] = 0;
                else outputs[i] = 1;
            }

            return outputs;
        }

        private double[] Compute(double[][] inputs)
        {
            double[] output = new double[inputs.Length];
            for (int i = 0; i < inputs.Length; i++) output[i] = Compute(inputs[i]);
            return output;
        }

        private double Compute(double[] inputs)
        {
            double output = Bias;
            for (int i = 0; i < inputs.Length; i++) output += inputs[i] * Weights[i];
            return output;
        }

        public double GetError(double[][] inputs, double[] desiredOutputs)
        {
            double[] outputs = Compute(inputs);

            double errorSum = 0;
            for (int i = 0; i < outputs.Length; i++)
            {
                errorSum += Math.Pow(ErrorFunc(outputs[i], desiredOutputs[i]), 2);
            }
            return errorSum / outputs.Length;
        }

        public double TrainORGate(double[][] inputs, double[] desiredOutputs, double currentError)
        {   
            Random rand = new Random();
            int chosenIndex = rand.Next(0, Weights.Length + 1);
            int valAlteration = rand.Next(0, 2) == 1 ? -1 : 1;
            double originalWeight = chosenIndex < Weights.Length ? Weights[chosenIndex] : 0;
            double originalBias = Bias;

            if (chosenIndex < Weights.Length) Weights[chosenIndex] += MutationAmount * valAlteration;
            else Bias += MutationAmount * valAlteration;

            double error = GetError(inputs, desiredOutputs);

            if (error < currentError) return error;
            if (chosenIndex < Weights.Length) Weights[chosenIndex] = originalWeight;
            else Bias = originalBias;

            return currentError;
        }
    }
}
