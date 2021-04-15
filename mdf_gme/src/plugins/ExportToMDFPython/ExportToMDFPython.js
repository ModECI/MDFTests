/*globals define*/
/*eslint-env node, browser*/

define([
    'text!./metadata.json',
    'mdf_gme/instance-converter',
    'webgme-json-importer/JSONImporter',
    'plugin/PluginBase',
], function (
    pluginMetadata,
    MDFConverter,
    JSONImporter,
    PluginBase
) {
    'use strict';

    pluginMetadata = JSON.parse(pluginMetadata);

    class ExportToMDFPython extends PluginBase {
        constructor() {
            super();
            this.pluginMetadata = pluginMetadata;
        }

        async main(callback) {
            const mdfJson = await this.getMDFJson(this.activeNode);
            console.log(mdfJson);
            // TODO: we need to inline this in a python file and create an EvaluableGraph
            // TODO: Then we need to assign the EvaluableGraph instance to "result"
        }

        async getMDFJson(node) {
            const importer = new JSONImporter(this.core, this.rootNode);
            const json = await importer.toJSON(this.activeNode);
            await this.setBasePtrsToMetaTag(json);
            return Object.fromEntries([MDFConverter.Model.toMDF(json)]);
        }

        async setBasePtrsToMetaTag(json) {
            const {base} = json.pointers;
            const baseNode = await this.core.loadByPath(this.rootNode, base);
            const metaTag = `@meta:${this.core.getAttribute(baseNode, 'name')}`;
            json.pointers.base = metaTag;

            if (json.children) {
                json.children = await Promise.all(
                    json.children.map(child => this.setBasePtrsToMetaTag(child))
                );
            }
            return json;
        }
    }

    ExportToMDFPython.metadata = pluginMetadata;

    return ExportToMDFPython;
});
