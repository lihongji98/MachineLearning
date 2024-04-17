set "src=no"
set "SCRIPTS=D:\nmt\mosesdecoder\scripts"
set "BPEROOT=D:\nmt\subword-nmt\subword_nmt"
set "data_dir=D:\pycharm_projects\MachineLearning\transformer"
set "model_dir=D:\pycharm_projects\MachineLearning\transformer\voc"

REM Define tools
set "TOKENIZER=%SCRIPTS%\tokenizer\tokenizer.perl"
set "TC=%SCRIPTS%\recaser\truecase.perl"
set "NORM_PUNC=%SCRIPTS%\tokenizer\normalize-punctuation.perl"

REM Run Perl commands
perl "%NORM_PUNC%" -l %src% < "%data_dir%\infer.%src%" > "%data_dir%\norm.%src%"
perl "%TOKENIZER%" -l %src% < "%data_dir%\norm.%src%" > "%data_dir%\norm_tok.%src%"
perl "%TC%" --model "%model_dir%\truecase-model.%src%" < "%data_dir%\norm_tok.%src%" > "%data_dir%\norm_tok_true.%src%"

python "%BPEROOT%\apply_bpe.py" -c "%model_dir%\bpecode.%src%" --vocabulary "%model_dir%\voc.%src%" < "%data_dir%\norm_tok_true.%src%" > "%data_dir%\infer.%src%"

del /F /Q "%data_dir%\norm.%src%"
del /F /Q "%data_dir%\norm_tok.%src%"
del /F /Q "%data_dir%\norm_tok_true.%src%"
