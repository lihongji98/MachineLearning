set "src=no"
set "SCRIPTS=D:\nmt\mosesdecoder\scripts"
set "BPEROOT=D:\nmt\subword-nmt\subword_nmt"
set "data_dir=D:\pycharm_projects\MachineLearning\transformer"
set "model_dir=D:\pycharm_projects\MachineLearning\transformer\voc"

REM Define tools
set "DETOKENIZER=%SCRIPTS%/tokenizer/detokenizer.perl"
set "DETC=%SCRIPTS%\recaser\detruecase.perl"

REM Run Perl commands
perl %DETC% < "%data_dir%\infer.%src%" > "%data_dir%\infer_det.%src%"
perl %DETOKENIZER% -l %src% < "%data_dir%/infer_det.%src%" > "%data_dir%/infer.%src%"

del /F /Q "%data_dir%\infer_det.%src%"
