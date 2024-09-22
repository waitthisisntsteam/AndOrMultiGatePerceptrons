namespace AndOrMultiGatePerceptrons
{
    internal class Program
    {
        static double Error(double output, double desiredOutput) => desiredOutput - output;
        static void Main(string[] args)
        {
            Perceptron gate = new Perceptron(3, 0.01, Error);
            gate.Randomize(new Random(), 0, 2);
            double[][] input = [ [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
            double[] desiredOutputs = [0, 1, 1, 1, 0, 0, 0, 1];

            double currentError = gate.GetError(input, desiredOutputs);
            while (true)
            {
                currentError = gate.TrainORGate(input, desiredOutputs, currentError);
                Console.WriteLine(currentError);

                var d = gate.Compute(input);
                ;
            }
        }
    }
}
