# -*- coding: utf8 -*-

from django.shortcuts import render, redirect
from django.template import RequestContext, loader
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from testing import test
from django.contrib import messages
from django.core.urlresolvers import reverse
import commands
import os
# Create your views here.

DIR = os.path.dirname(os.path.abspath(__file__))
def home(request):
	return render(request,'home.html')

def getSarcasm(request):
	sent = request.GET.get('sent')
	lan = request.GET.get('lan')

	#MT
	t = sent.replace("\'","").replace("\"","")
	print t
	cmd = DIR+'/trans '+lan+':en -b " '+t+'"'
	print cmd.encode("utf-8",'ignore')
	status, outtemp = commands.getstatusoutput(cmd.encode("utf8",'ignore'))
	print status,outtemp
	a = test(outtemp)

	messages.warning(request,"Using MT : ")
	M = "      English translation of '"+sent+ "' using MT is '"+outtemp+"'. "
	messages.warning(request,M)
	messages.warning(request, "     Given input: '"+sent+"' is "+ a+".")	

	t = sent.replace("\'","").replace("\"","").replace("\n","").replace("\t","").split(" ")
	newsentence = ""
	for part in t:
		print part
		cmd = DIR+'/trans '+lan+':en -b " '+part+'"'
		status, outtemp = commands.getstatusoutput(cmd.encode("utf8",'ignore'))
		print status,outtemp
		newsentence = newsentence + " "+outtemp
	a = test(newsentence)
	M = "     English translation of '"+sent+ "' using Bilingual Mapping is '"+newsentence+"'."

	messages.warning(request,"  ")
	messages.warning(request,"Using Bilingual Mapping : ")
	messages.warning(request,M)
	messages.warning(request, "     Given input: '"+sent+"' is "+ a+".")	

	return redirect("/home")
