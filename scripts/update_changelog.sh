LAST_TAG=`git tag --sort version:refname | tail -n 1`
echo "Changes since $LAST_TAG:" >> $CHANGELOG
git log --pretty="[%cs] %h - %s" ${LAST_TAG}..HEAD
