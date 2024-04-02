@echo off

echo Do you want to run this script? (Y/N)
choice /c yn /m "Enter Y or N"

if errorlevel 2 goto :eof

REM Define paths
set "SCRIPTS=D:\nmt\mosesdecoder\scripts"
set "BPEROOT=D:\nmt\subword-nmt\subword_nmt"

REM Define directories
set "data_dir=D:\pycharm_projects\MachineLearning\transformer\data"
set "model_dir=D:\pycharm_projects\MachineLearning\transformer\model"

REM Define tools
set "TOKENIZER=%SCRIPTS%\tokenizer\tokenizer.perl"
set "TRAIN_TC=%SCRIPTS%\recaser\train-truecaser.perl"
set "TC=%SCRIPTS%\recaser\truecase.perl"
set "NORM_PUNC=%SCRIPTS%\tokenizer\normalize-punctuation.perl"
set "CLEAN=%SCRIPTS%\training\clean-corpus-n.perl"

REM Run Perl commands
echo NORM_PUNCTUALIZING...
perl "%NORM_PUNC%" -l en < "%data_dir%\raw.en" > "%data_dir%\norm.en"
perl "%NORM_PUNC%" -l en < "%data_dir%\raw.no" > "%data_dir%\norm.no"

echo TOKENIZING...
perl "%TOKENIZER%" -l en < "%data_dir%\norm.en" > "%data_dir%\norm_tok.en"
perl "%TOKENIZER%" -l no < "%data_dir%\norm.no" > "%data_dir%\norm_tok.no"

del /F /Q "%data_dir%\norm.en"
del /F /Q "%data_dir%\norm.no"

echo TRANING AND TRUECASING...
perl "%TRAIN_TC%" --model "%model_dir%\truecase-model.en" --corpus "%data_dir%\norm_tok.en"
perl "%TC%" --model "%model_dir%\truecase-model.en" < "%data_dir%\norm_tok.en" > "%data_dir%\norm_tok_true.en"

echo TRANING AND TRUECASING...
perl "%TRAIN_TC%" --model "%model_dir%\truecase-model.no" --corpus "%data_dir%\norm_tok.no"
perl "%TC%" --model "%model_dir%\truecase-model.no" < "%data_dir%\norm_tok.no" > "%data_dir%\norm_tok_true.no"

del /F /Q "%data_dir%\norm_tok.en"
del /F /Q "%data_dir%\norm_tok.no"

REM Run Python commands
echo TRAINING BPE TOKENIZER...
python "%BPEROOT%\learn_joint_bpe_and_vocab.py" --input "%data_dir%\norm_tok_true.en" -s 8000 -o "%model_dir%\bpecode.en" --write-vocabulary "%model_dir%\voc.en"
python "%BPEROOT%\apply_bpe.py" -c "%model_dir%\bpecode.en" --vocabulary "%model_dir%\voc.en" < "%data_dir%\norm_tok_true.en" > "%data_dir%\norm_tok_true_bpe.en"

python "%BPEROOT%\learn_joint_bpe_and_vocab.py" --input "%data_dir%\norm_tok_true.no" -s 8000 -o "%model_dir%\bpecode.no" --write-vocabulary "%model_dir%\voc.no"
python "%BPEROOT%\apply_bpe.py" -c "%model_dir%\bpecode.no" --vocabulary "%model_dir%\voc.no" < "%data_dir%\norm_tok_true.no" > "%data_dir%\norm_tok_true_bpe.no"

del /F /Q "%data_dir%\norm_tok_true.en"
del /F /Q "%data_dir%\norm_tok_true.no"

del /F /Q "%data_dir%\truecase-model.en"
del /F /Q "%data_dir%\truecase-model.no"

REM Rename or move the files
move "%model_dir%\norm_tok_true_bpe.en" "%data_dir%\toclean.en"
move "%data_dir%\norm_tok_true_bpe.no" "%data_dir%\toclean.no"

REM Clean the files using the CLEAN script
perl "%CLEAN%" "%data_dir%\toclean" "no" "en" "%data_dir%\clean" 1 128

del /F /Q "%data_dir%\toclean.en"
del /F /Q "%data_dir%\toclean.no"

del /F /Q "%model_dir%\bpecode.en"
del /F /Q "%model_dir%\bpecode.no"
