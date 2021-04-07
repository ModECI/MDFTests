# MDF in WebGME
This contains a tool for converting the [MDF specification](https://github.com/ModECI/MDF/blob/documentation/docs/MDF_specification.json) into JSON compatible with [JSON importer](https://github.com/deepforge-dev/webgme-json-importer/tree/master/src/common). This allows us to programmatically create a metamodel and, as a result, use WebGME as a design environment for MDF.

## Quick Start
### Loading the spec into WebGME
First, install dependencies with `npm install`. Then convert the MDF specification using
```
node spec_to_gme.js path/to/MDF/spec.json
```

Finally, import the JSON into WebGME just like the [examples](https://github.com/deepforge-dev/webgme-json-importer/tree/master/examples) (suffixed with "\_meta")!

### Loading instances into WebGME
```
node instance_to_gme.js path/to/MDF/instance.json
```
