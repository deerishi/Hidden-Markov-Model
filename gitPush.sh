#! /bin/bash


git pull -u origin master
git add .
git commit -m "first commit"
git config credential.helper 'store'
git push -u origin master



