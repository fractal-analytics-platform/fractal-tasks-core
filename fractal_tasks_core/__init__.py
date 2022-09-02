__FRACTAL_MANIFEST__ = [
      {
          "resource_type": "core task",
          "name": "dummy",
          "module": f"{__name__}.dummy:dummy",
          "input_type": "Any",
          "output_type": "None",
          "default_args": {
              "message": "dummy default",
              "index": 0,
              "executor": "cpu",
          },
      },
]
