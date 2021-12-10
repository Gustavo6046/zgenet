enum NeuronType {
    NEU_ZERO, // ignore when set, 0 when get
    NEU_INPUTS,
    NEU_OUTPUTS,
    NEU_HIDDEN
};

enum MutateType {
    MT_CONNVALUES,
    MT_NEWCONN,
    MT_KILLCONN
};

class NNAgent : Actor abstract {
    abstract void refreshInputs(NeuralNetwork net);
    abstract void applyOutputs(NeuralNetwork net);
    abstract Actor getActor();
    abstract void setNet(NeuralNetwork newNet);

    abstract NNAgent reproduce();
}

class Connection {
    // Merely holds a connection from neuron to neuron.
    // Can't be struct because we have to store in an array.
    int from, to;
    int weight, offset;

    static Connection Make(int from, int to, int weight, int offset) {
        Connection conn;
        conn = Connection(new('Connection'));

        conn.from = from;
        conn.to = to;
        conn.weight = weight;
        conn.offset = offset;

        return conn;
    }

    Connection duplicate() {
        return Connection.Make(from, to, weight, offset);
    }

    void mutateWeights(float amount) {
        weight += FRandom(-amount, amount) * NeuralNetwork.FIXUNIT;
        offset += FRandom(-amount, amount) * NeuralNetwork.FIXUNIT;
    }
}

// Neuron index:
// - 2 least significant bits is a NeuronType
// - right shifted by 2 is actual array index
class NeuralNetwork {
    int num_inputs, num_outputs;
    int num_hidden;
    
    Array<int> inputs, outputs, hidden;
    Array<int> newHidden;
    Array<Connection> connections;
    
    NNAgent agent;
    
    const FIXUNIT = 32768;
    
    static NeuralNetwork Make(int ninputs, int noutputs, int nhidden, NNAgent agent) {
        NeuralNetwork nn = NeuralNetwork(new('NeuralNetwork'));
        
        nn.num_inputs = ninputs;
        nn.num_outputs = noutputs;
        nn.num_hidden = nhidden;
        
        int i;
        
        for (i = 0; i < ninputs; i++) {
            nn.inputs.Push(0);
        }
        
        for (i = 0; i < noutputs; i++) {
            nn.outputs.Push(0);
        }
        
        for (i = 0; i < nhidden; i++) {
            nn.hidden.Push(0);
        }
        
        nn.agent = agent;
        
        return nn;
    }
    
    void neuronGet(int ref, out int value) {
        int index = ref >> 2;
        NeuronType type = ref & 0x3;
        
        switch (type) {
            case NEU_INPUTS:
                value = inputs[index];
                break;
            
            case NEU_OUTPUTS:
                value = outputs[index];
                break;
                
            case NEU_HIDDEN:
                value = hidden[index];
                break;
            
            case NEU_ZERO:
                value = 0;
                break;
        }
    }
    
    void neuronSet(int ref, int newValue) {
        int index = ref >> 2;
        NeuronType type = ref & 0x3;
        
        switch (type) {
            case NEU_INPUTS:
                inputs[index] = newValue;
                break;
            
            case NEU_OUTPUTS:
                outputs[index] = newValue;
                break;
                
            case NEU_HIDDEN:
                hidden[index] = newValue;
                break;
            
            case NEU_ZERO:
                break;
        }
    }
    
    void neuronAdd(int ref, int addValue) {
        int index = ref >> 2;
        NeuronType type = ref & 0x3;
        
        switch (type) {
            case NEU_INPUTS:
                inputs[index] += addValue;
                break;
            
            case NEU_OUTPUTS:
                outputs[index] += addValue;
                break;
                
            case NEU_HIDDEN:
                hidden[index] += addValue;
                break;
            
            case NEU_ZERO:
                break;
        }
    }

    void neuronAddTemp(int ref, int addValue) {
        int index = ref >> 2;
        NeuronType type = ref & 0x3;
        
        switch (type) {
            case NEU_OUTPUTS:
                outputs[index] += addValue;
                break;
                
            case NEU_HIDDEN:
                newHidden[index] += addValue;
                break;
        }
    }
    
    Connection connect(int from, int to) {
        NeuronType fromType = from & 0x3;
        NeuronType toType = to & 0x3;
        
        if (fromType == NEU_ZERO || toType == NEU_ZERO || toType == NEU_INPUTS || fromType == NEU_OUTPUTS) {
            return null;
        }

        Connection newConn = Connection.Make(from, to, FRandom(-4, 4) * FIXUNIT, FRandom(-4, 4) * FIXUNIT);
        connections.Push(newConn);

        return newConn;
    }

    Connection connectWeight(int from, int to, int weight, int offset) {
        NeuronType fromType = from & 0x3;
        NeuronType toType = to & 0x3;
        
        if (fromType == NEU_ZERO || toType == NEU_ZERO || toType == NEU_INPUTS || fromType == NEU_OUTPUTS) {
            return null;
        }

        Connection newConn = Connection.Make(from, to, weight, offset);
        connections.Push(newConn);

        return newConn;
    }
    
    void processValues() {
        newHidden.Copy(hidden);

        for (int oi = 0; oi < num_outputs; oi++) {
            outputs[oi] = 0;
        }
    
        Connection conn;
        int ci, val;
    
        for (ci = 0; ci < connections.Size(); ci++) {
            conn = connections[ci];
            
            neuronGet(conn.from, val);
            val = val * conn.weight / FIXUNIT + conn.offset;

            if (val < 0) {
                // asymmetry is fun!
                val /= 4;
            }
            
            neuronAddTemp(conn.to, val * FIXUNIT / (FIXUNIT + abs(val)));
        }

        hidden.move(newHidden);
    }
    
    void update() {
        if (agent.getActor() == null) {
            Destroy();
        }
    
        agent.refreshInputs(self);
        
        processValues();
        
        agent.applyOutputs(self);
    }

    override void OnDestroy() {
        int i;
        
        for (i = 0; i < connections.Size(); i++) {
            connections[i].Destroy();
        }

        connections.Clear();
    }

    Connection connectRandom() {
        int fromType = (Random(0, 1) == 0) ? NEU_INPUTS : NEU_HIDDEN;
        int toType = (Random(0, 2) == 0) ? NEU_OUTPUTS : NEU_HIDDEN;
        
        int fromIdx, toIdx;

        switch (fromType) {
            case NEU_INPUTS:	fromIdx = Random(0, num_inputs - 1);	break;
            case NEU_HIDDEN:	fromIdx = Random(0, num_hidden - 1);	break;
        }

        switch (toType) {
            case NEU_OUTPUTS:	toIdx = Random(0, num_outputs - 1);	break;
            case NEU_HIDDEN:	toIdx = Random(0, num_hidden - 1);	break;
        }

        return connect(fromIdx << 2 | fromType, toIdx << 2 | toType);
    }

    NeuralNetwork replicate(float netMutateChance = 25, float connMutateAmount = 0.6) {
        bool bMutate = FRandom(0, 100) < netMutateChance;

        // Make a duplicate of this network and its agent.
        NNAgent newAgent = agent.reproduce();

        if (newAgent == null) {
            return null;
        }

        NeuralNetwork newNet = NeuralNetwork.Make(num_inputs, num_outputs, num_hidden, newAgent);

        int ci;

        for (ci = 0; ci < connections.Size(); ci++) {
            newNet.connections.Push(connections[ci].duplicate());
        }

        if (!bMutate) {
            newAgent.setNet(newNet);
            return newNet;
        }

        // Handle mutation.
        MutateType mtype = Random(0, 3); // MNT: update whenever MutateType is changed
        int which;

        switch (mtype) {
            case MT_CONNVALUES:
                which = Random(0, newNet.connections.Size() - 1);

                newNet.connections[which].mutateWeights(connMutateAmount);

                break;

            case MT_NEWCONN:
                newNet.connectRandom();

                break;

            case MT_KILLCONN:
                which = Random(0, newNet.connections.Size() - 1);

                newNet.connections.Delete(which);

                break;
        }

        newAgent.setNet(newNet);

        return newNet;
    }
}
