#!/bin/bash

echo matlab -nodisplay -nojvm  -r "blacken_block $1; quit;"

matlab -nodisplay -nojvm  -r "blacken_block $1; quit;"
