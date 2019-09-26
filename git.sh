git add -A .

if [ $# -eq 1 ]; then
  git commit -m $1
else
  git commit -m "update"
fi

git push origin master
