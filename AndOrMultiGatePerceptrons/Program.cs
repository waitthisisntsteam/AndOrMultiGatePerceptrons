namespace AndOrMultiGatePerceptrons
{
    internal class Program
    {
        static double Error(double output, double desiredOutput) => desiredOutput - output;
        static void Main(string[] args)
        {
            Perceptron orGate = new Perceptron(2, 0.01, Error);
            orGate.Randomize(new Random(), 0, 2);
            double[][] input = [ [0,0], [1,0], [0, 1], [1, 1] ];
            double[] desiredOutputs = [0, 1, 1, 1];

            double currentError = orGate.GetError(input, desiredOutputs);
            while (true)
            {
                currentError = orGate.TrainORGate(input, desiredOutputs, currentError);
                Console.WriteLine(currentError);

                var d = orGate.Compute(input);
                ;
            }
        }
    }
}
