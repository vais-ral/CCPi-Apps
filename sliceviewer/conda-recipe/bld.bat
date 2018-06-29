IF NOT DEFINED CIL_VERSION (
ECHO CIL_VERSION Not Defined.
exit 1
)

mkdir "%SRC_DIR%\sliceviewer"
ROBOCOPY /E "%RECIPE_DIR%\..\.." "%SRC_DIR%"
cd %SRC_DIR%\sliceviewer

:: issue cmake to create setup.py
::cmake . 

%PYTHON% setup.py build_ext
if errorlevel 1 exit 1
%PYTHON% setup.py install
if errorlevel 1 exit 1
