
CHANGELOG=CHANGELOG
TMP=tmp_$CHANGELOG

cat $CHANGELOG > $TMP

LAST_TAG=`git tag --sort version:refname | tail -n 2 | head -n 1`

date > $CHANGELOG
echo "Changes since $LAST_TAG:" >> $CHANGELOG
git log --pretty="[%cs] %h - %s" ${PREVIOUS}..HEAD >> $CHANGELOG
echo >> $CHANGELOG

cat $TMP >> $CHANGELOG
rm $TMP
