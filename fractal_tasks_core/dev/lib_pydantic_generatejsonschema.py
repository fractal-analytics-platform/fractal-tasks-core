"""
Custom Pydantic v2 JSON Schema generation tools.

As of Pydantic V2, the JSON Schema representation of model attributes marked
as `Optional` changed, and the new behavior consists in marking the
corresponding properties as an `anyOf` of either a `null` or the actual type.
This is not always the required behavior, see e.g.
* https://github.com/pydantic/pydantic/issues/7161
* https://github.com/pydantic/pydantic/issues/8394

Here we list some alternative ways of reverting this change.
"""
import logging

from pydantic.json_schema import GenerateJsonSchema
from pydantic.json_schema import JsonSchemaValue
from pydantic_core.core_schema import WithDefaultSchema

logger = logging.getLogger("CustomGenerateJsonSchema")


class CustomGenerateJsonSchema(GenerateJsonSchema):
    def get_flattened_anyof(
        self, schemas: list[JsonSchemaValue]
    ) -> JsonSchemaValue:
        null_schema = {"type": "null"}
        if null_schema in schemas:
            logger.warning(
                "Drop `null_schema` before calling `get_flattened_anyof`"
            )
            schemas.pop(schemas.index(null_schema))
        return super().get_flattened_anyof(schemas)

    def default_schema(self, schema: WithDefaultSchema) -> JsonSchemaValue:
        json_schema = super().default_schema(schema)
        if "default" in json_schema.keys() and json_schema["default"] is None:
            logger.warning(f"Pop `None` default value from {json_schema=}")
            json_schema.pop("default")
        return json_schema


# class GenerateJsonSchemaA(GenerateJsonSchema):
#     def nullable_schema(self, schema):
#         null_schema = {"type": "null"}
#         inner_json_schema = self.generate_inner(schema["schema"])
#         if inner_json_schema == null_schema:
#             return null_schema
#         else:
#             logging.info("A: Skip calling `get_flattened_anyof` method")
#             return inner_json_schema


# class GenerateJsonSchemaB(GenerateJsonSchemaA):
#     def default_schema(self, schema: WithDefaultSchema) -> JsonSchemaValue:
#         original_json_schema = super().default_schema(schema)
#         new_json_schema = deepcopy(original_json_schema)
#         default = new_json_schema.get("default", None)
#         if default is None:
#             logging.info("B: Pop None default")
#             new_json_schema.pop("default")
#         return new_json_schema


# class GenerateJsonSchemaC(GenerateJsonSchema):
#     def get_flattened_anyof(
#         self, schemas: list[JsonSchemaValue]
#     ) -> JsonSchemaValue:

#         original_json_schema_value = super().get_flattened_anyof(schemas)
#         members = original_json_schema_value.get("anyOf")
#         logging.info("C", original_json_schema_value)
#         if (
#             members is not None
#             and len(members) == 2
#             and {"type": "null"} in members
#         ):
#             new_json_schema_value = {"type": [t["type"] for t in members]}
#             logging.info("C", new_json_schema_value)
#             return new_json_schema_value
#         else:
#             return original_json_schema_value
