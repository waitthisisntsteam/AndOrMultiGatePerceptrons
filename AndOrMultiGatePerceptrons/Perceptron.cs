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

        public double Compute(double[] inputs)
        {
            double output = Bias;
            for (int i = 0; i < inputs.Length; i++) output += inputs[i] * Weights[i];
            return output;
        }

        public double[] Compute(double[][] inputs)
        {
            double[] output = new double[inputs.Length];
            for (int i = 0; i < inputs.Length; i++) output[i] = Compute(inputs[i]);
            return output;
        }

        public double GetError(double[][] inputs, double[] desiredOutputs)
        {
            double[] outputs = Compute(inputs);

            double errorSum = 0;
            for (int i = 0; i < outputs.Length; i++) errorSum += ErrorFunc(outputs[i], desiredOutputs[i]);
            return errorSum / outputs.Length;
        }

        public double TrainORGate(double[][] inputs, double[] desiredOutputs, double currentError)
        {   
            Random rand = new Random();
            for (int i = 0; i < inputs[0].Length; i++)
            {
                for (int j = 0; j < inputs[1].Length; j++)
                {
                    inputs[i][j] += MutationAmount;
                }
            }

            double error = GetError(inputs, desiredOutputs);

            if (error < currentError) return error;
            return currentError;
        }
    }
}
