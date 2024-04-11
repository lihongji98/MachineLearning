@echo off

echo Do you want to run this script? (Y/N)
choice /c yn /m "Enter Y or N"

if errorlevel 2 goto :eof

REM Define lang
set "src=no"
set "trg=en"

REM Define vocabulary size
set "vocab_size=8000"

REM Define paths
set "SCRIPTS=D:\nmt\mosesdecoder\scripts"
set "BPEROOT=D:\nmt\subword-nmt\subword_nmt"

REM Define directories
set "data_dir=D:\pycharm_projects\MachineLearning\transformer\data"
set "model_dir=D:\pycharm_projects\MachineLearning\transformer\voc"

REM Define tools
set "TOKENIZER=%SCRIPTS%\tokenizer\tokenizer.perl"
set "TRAIN_TC=%SCRIPTS%\recaser\train-truecaser.perl"
set "TC=%SCRIPTS%\recaser\truecase.perl"
set "NORM_PUNC=%SCRIPTS%\tokenizer\normalize-punctuation.perl"
set "CLEAN=%SCRIPTS%\training\clean-corpus-n.perl"

REM Run Perl commands
echo NORM_PUNCTUALIZING...
perl "%NORM_PUNC%" -l %src% < "%data_dir%\raw.%src%" > "%data_dir%\norm.%src%"
perl "%NORM_PUNC%" -l %trg% < "%data_dir%\raw.%trg%" > "%data_dir%\norm.%trg%"

echo TOKENIZING...
perl "%TOKENIZER%" -l %src% < "%data_dir%\norm.%src%" > "%data_dir%\norm_tok.%src%"
perl "%TOKENIZER%" -l %trg% < "%data_dir%\norm.%trg%" > "%data_dir%\norm_tok.%trg%"

del /F /Q "%data_dir%\norm.%trg%"
del /F /Q "%data_dir%\norm.%src%"

echo TRANING AND TRUECASING...
perl "%TRAIN_TC%" --model "%model_dir%\truecase-model.%trg%" --corpus "%data_dir%\norm_tok.%trg%"
perl "%TC%" --model "%model_dir%\truecase-model.%trg%" < "%data_dir%\norm_tok.%trg%" > "%data_dir%\norm_tok_true.%trg%"

echo TRANING AND TRUECASING...
perl "%TRAIN_TC%" --model "%model_dir%\truecase-model.%src%" --corpus "%data_dir%\norm_tok.%src%"
perl "%TC%" --model "%model_dir%\truecase-model.%src%" < "%data_dir%\norm_tok.%src%" > "%data_dir%\norm_tok_true.%src%"

del /F /Q "%data_dir%\norm_tok.%trg%"
del /F /Q "%data_dir%\norm_tok.%src%"


REM Run Python commands
echo TRAINING BPE TOKENIZER...
python "%BPEROOT%\learn_joint_bpe_and_vocab.py" --input "%data_dir%\norm_tok_true.%trg%" -s %vocab_size% -o "%model_dir%\bpecode.%trg%" --write-vocabulary "%model_dir%\voc.%trg%"
python "%BPEROOT%\apply_bpe.py" -c "%model_dir%\bpecode.%trg%" --vocabulary "%model_dir%\voc.%trg%" < "%data_dir%\norm_tok_true.%trg%" > "%data_dir%\norm_tok_true_bpe.%trg%"

python "%BPEROOT%\learn_joint_bpe_and_vocab.py" --input "%data_dir%\norm_tok_true.%src%" -s %vocab_size% -o "%model_dir%\bpecode.%src%" --write-vocabulary "%model_dir%\voc.%src%"
python "%BPEROOT%\apply_bpe.py" -c "%model_dir%\bpecode.%src%" --vocabulary "%model_dir%\voc.%src%" < "%data_dir%\norm_tok_true.%src%" > "%data_dir%\norm_tok_true_bpe.%src%"

del /F /Q "%data_dir%\norm_tok_true.%trg%"
del /F /Q "%data_dir%\norm_tok_true.%src%"

del /F /Q "%model_dir%\truecase-model.%trg%"
del /F /Q "%model_dir%\truecase-model.%src%"

REM Rename or move the files
move "%data_dir%\norm_tok_true_bpe.%trg%" "%data_dir%\toclean.%trg%"
move "%data_dir%\norm_tok_true_bpe.%src%" "%data_dir%\toclean.%src%"

REM Clean the files using the CLEAN script
perl "%CLEAN%" "%data_dir%\toclean" "%src%" "%trg%" "%data_dir%\clean" 1 126

del /F /Q "%data_dir%\toclean.%trg%"
del /F /Q "%data_dir%\toclean.%src%"

del /F /Q "%model_dir%\bpecode.%trg%"
del /F /Q "%model_dir%\bpecode.%src%"