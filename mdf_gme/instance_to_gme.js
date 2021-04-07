const json = require(process.argv[2]);

function parseModel(name, data) {
    const {format, generating_application} = data;
    const children = Object.entries(data.graphs).map(entry => parseGraph(...entry));
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

function parseGraph(name, data) {
    const children = Object.entries(data.nodes).map(entry => parseNode(...entry))
        .concat(Object.entries(data.edges).map(entry => parseEdge(...entry)))
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

function parseNode(name, data) {
    const parameters = Object.entries(data.parameters).map(entry => parseDictEntry(name, ...entry));

    const inputNodes = Object.entries(data.input_ports || {}).map(entry => parseInputPort(name, ...entry));
    const outputNodes = Object.entries(data.output_ports || {}).map(entry => parseOutputPort(name, ...entry));
    const functions = Object.entries(data.functions || {}).map(entry => parseFunction(...entry));
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

function parseDictEntry(parentName, name, data) {
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

function parseInputPort(nodeName, name, data) {
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

function parseOutputPort(nodeName, name, data) {
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

function parseFunction(name, data) {
    const args = Object.entries(data.args || {}).map(entry => parseDictEntry(name, ...entry));
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

function parseEdge(name, data) {
    return {
        attributes: {name},
        pointers: {
            base: '@meta:Edge',
            src: `@id:${data.sender}_${data.sender_port}`,
            dst: `@id:${data.receiver}_${data.receiver_port}`,
        }
    };
}

const output = Object.entries(json).map(entry => parseModel(...entry)).shift();
console.log(JSON.stringify(output, null, 2));
