all: en cs it-mt-cs

en : en-bl en-mt
it : it-bl-en it-bl-cs it-mt-en it-mt-cs
cs : cs-bl cs-mt 


it-bl-en:
	python training.py lan-it/BD/en-it lan-it/BD/en-it/models/model 20 notest 1000 italian
	python testing.py lan-it/BD/it-it lan-it/BD/en-it/models/model lan-it/BD/results/en-it_results.txt test 3130 italian

it-bl-cs:
	python training.py lan-it/BD/cs-it lan-it/BD/cs-it/models/model 10 test 274 italian
	python testing.py lan-it/BD/it-it lan-it/BD/cs-it/models/model lan-it/BD/results/cs-it_results.txt test 3130 italian

it-mt-en:
	python training.py lan-it/MT/en-it lan-it/MT/en-it/models/model 20 notest 1000 italian
	python testing.py lan-it/MT/it-it lan-it/MT/en-it/models/model lan-it/MT/results/en-it_results.txt test 3130 italian

it-mt-cs:
	python training.py lan-it/MT/cs-it lan-it/MT/cs-it/models/model 10 test 274 italian
	python testing.py lan-it/MT/it-it lan-it/MT/cs-it/models/model lan-it/MT/results/cs-it_results.txt test 3130 italian


en-bl:
	# python training.py lan-en/BD/en-en lan-en/BD/en-en/models/model 20 notest 1000 english
	python testing.py lan-en/BD/it-en lan-en/BD/en-en/models/model lan-en/BD/results/results.txt test 3130 english

en-mt:
	python testing.py lan-en/MT/it-en lan-en/BD/en-en/models/model lan-en/MT/results/results.txt test 3130 english


cs-bl:
	python training.py lan-cs/BD/cs-cs lan-cs/BD/cs-cs/models/model 10 test 274 czech
	python testing.py lan-cs/BD/it-cs lan-cs/BD/cs-cs/models/model lan-cs/BD/results/results.txt test 3130 czech

cs-mt:
	python testing.py lan-cs/MT/it-cs lan-cs/BD/cs-cs/models/model lan-cs/MT/results/results.txt test 3130 czech
	


