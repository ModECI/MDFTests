const Model = {
    toMDF() {
        // TODO
    },

    toGME(name, data) {
        const {format, generating_application} = data;
        const children = Object.entries(data.graphs).map(entry => Graph.toGME(...entry));
        return {
            attributes: {
                name,
                id: name,
                format,
                generating_application,
                notes: data.notes || '',
            },
            pointers: {
                base: '@meta:Model',
            },
            children,
        };
    }
};

const Graph = {
    toMDF() {
        // TODO
    },

    toGME(name, data) {
        const children = Object.entries(data.nodes).map(entry => Node.toGME(...entry))
            .concat(Object.entries(data.edges).map(entry => Edge.toGME(...entry)))
        return {
            attributes: {
                id: name,
                name,
                notes: data.notes || '',
            },
            pointers: {
                base: '@meta:Graph',
            },
            children,
        };
    }
};

const Node = {
    toMDF() {
        // TODO
    },

    toGME(name, data) {
        const parameters = Object.entries(data.parameters).map(entry => Parameters.toGME(name, ...entry));

        const inputNodes = Object.entries(data.input_ports || {}).map(entry => InputPort.toGME(name, ...entry));
        const outputNodes = Object.entries(data.output_ports || {}).map(entry => OutputPort.toGME(name, ...entry));
        const functions = Object.entries(data.functions || {}).map(entry => Function.toGME(...entry));
        return {
            id: `@id:${name}`,
            attributes: {
                name,
                id: name,
            },
            pointers: {
                base: '@meta:Model'
            },
            sets: {
                parameters: parameters.map(param => param.id)
            },
            children: parameters.concat(inputNodes, outputNodes, functions)
        };
    }
};

const Parameters = {
    toMDF() {
        // TODO
    },
    
    toGME(parentName, name, data) {
        return {
            id: `@id:${parentName}_${name}`,
            attributes: {
                name,
                value: data
            },
            pointers: {
                base: '@meta:DictionaryEntry'
            }
        };
    }
};

const InputPort = {
    toMDF() {
        // TODO
    },

    toGME(nodeName, name, data) {
        return {
            id: `@id:${nodeName}_${name}`,
            attributes: {
                name,
                id: name,
                shape: data.shape,
            },
            pointers: {
                base: '@meta:InputPort'
            },
        };
    }
};

const OutputPort = {
    toMDF() {
        // TODO
    },

    toGME(nodeName, name, data) {
        return {
            id: `@id:${nodeName}_${name}`,
            attributes: {
                name,
                id: name,
                value: data.value,
            },
            pointers: {
                base: '@meta:OutputPort'
            }
        };
    }
};

const Function = {
    toMDF() {
        // TODO
    },

    toGME(name, data) {
        const args = Object.entries(data.args || {}).map(entry => Parameters.toGME(name, ...entry));
        return {
            attributes: {
                name,
                id: name,
                function: data.function,
                notes: data.notes || '',
            },
            pointers: {
                base: '@meta:Function',
            },
            sets: {
                args: args.map(arg => arg.id),
            },
            children: args,
        };
    }
};

const Edge = {
    toMDF() {
        // TODO
    },

    toGME(name, data) {
        return {
            attributes: {name},
            pointers: {
                base: '@meta:Edge',
                src: `@id:${data.sender}_${data.sender_port}`,
                dst: `@id:${data.receiver}_${data.receiver_port}`,
            }
        };
    }
};

const json = require(process.argv[2]);
const output = Object.entries(json).map(entry => Model.toGME(...entry)).shift();
console.log(JSON.stringify(output, null, 2));
