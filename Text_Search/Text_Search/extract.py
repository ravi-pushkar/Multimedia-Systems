import wikipedia


# Wikipedia.page returns errors a lot of times...
# Ex: 'Taylor Swift', 'Dua Lipa', 'Elon Musk' does not work but
#'TaylorSwift', 'DuaLipa', 'ElonMusk' does work





def getArticle(articleName):
    try:
        return wikipedia.page(articleName)
    except:
        return None


def getText(article):
    try:
        if isinstance(article, str):
            return getArticle(article).content
        else:
            return article.content
    except:
        return None








