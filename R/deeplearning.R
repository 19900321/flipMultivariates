#' Takes a data frame as input as converts output to a numeric matrix
#' Unlike data.matrix, factors are converted to an indicator matrix
#' We don't want to remove missing levels from prediction data
#' @importFrom flipTransformations FactorToIndicators
dataToNumeric <- function(df)
{
    ctmp <- unlist(lapply(df, function(x){paste(class(x), collapse=" ")}))
    ind <- which(ctmp == "character")
    if (length(ind) > 0)
    {
        for (i in ind)
        {
            warnings("Variable ", colnames(df)[i], " converted to a factor\n")
            df[,i] <- factor(df[,i])
        }
    }

    res <- as.data.frame(lapply(df, function(x){
                   if (any(class(x) == "factor"))
                   {
                       return(FactorToNumeric(x, remove.first=F))
                   } else
                   {
                       return(x)
                   }
    }))

    cnames <- c()
    for (i in 1:ncol(df))
    {
        if (any(class(df[,i]) == "factor"))
        {
            # factors are matched using colnames and label names
            # so we do not need to worry about underlying numeric value
            tmp <- sprintf("%s:%s", colnames(df)[i], levels(df[,i]))
            cnames <- c(cnames, tmp)
        } else
        {
            cnames <- c(cnames, colnames(df)[i])
        }
    }
    colnames(res) <- cnames
    return(res)
}


#' \code{DeepLearning}
#'
#' @description Constructs a deep learning model using a fully connected neural network with hidden layers
#' @param formula A formula of the form \code{groups ~ x1 + x2 + ...}
#' That is, the response is the grouping factor and the right hand side
#' specifies the (non-factor) discriminators, and any transformations, interactions,
#' or other non-additive operators will be ignored.
#' transformations nor
#' @param hidden Vector specifying the number of nodes in each hidden layer.
#' For example \code{hidden = c(50,25)} creates a neural network with 2 layers,
#' containing 50 nodes in the first layer and 25 nodes in the second layer.
#' @param unit.function Activation function of nodes in neural network. Can be one of
#' \code{"relu"} (rectified linear unit), \code{"sigmoid"}, \code{"softrelu"}
#' or \code{"tanh"}
#' @param optimizer Algorithm for adaptive learning rate used to train neural network
#' @param epochs Length of training process.
#' @param batch.size Size of training samples used in each weight-update iteration
#' @param data A \code{\link{data.frame}} from which variables specified
#' in formula are preferentially to be taken.
#' @param subset An optional vector specifying a subset of observations to be
#'   used in the fitting process, or, the name of a variable in \code{data}. It
#'   may not be an expression. \code{subset} may not
#' @param weights An optional vector of sampling weights, or, the name or, the
#'   name of a variable in \code{data}. It may not be an expression.
#' @param output One of \code{"Summary", "Training error"}, or \code{"Text"}.
#' @param missing How missing data is to be treated in the regression. Options:
#'   \code{"Error if missing data"},
#'   \code{"Exclude cases with missing data"},
#' @param seed The random number seed used in imputation.
#' @param show.labels Shows the variable labels, as opposed to the labels, in the outputs, where a
#' variables label is an attribute (e.g., attr(foo, "label")).
#' @param ... Other arguments to be supplied to \code{\link{darch}}.
#' @importFrom flipData GetData CleanSubset CleanWeights EstimationData DataFormula
#' @importFrom flipFormat Labels
#' @importFrom flipU OutcomeName
#' @importFrom flipTransformations AdjustDataToReflectWeights FactorToNumeric
#' @import mxnet
#' @examples
#' data(iris)
#' m.iris <- DeepLearning(Species~Sepal.Length+Sepal.Width+Petal.Length+Petal.Width, data=iris)
#' @export
DeepLearning <- function(formula,
                hidden = c(50, 50),
                unit.function = c("relu", "sigmoid", "softrelu", "tanh")[1],
                optimizer = c("adam", "rmsprop", "adadelta", "adagrad")[1],
                epochs = 500,
                batch.size = 100,
                data = NULL,
                subset = NULL,
                weights = NULL,
                output = c("Summary", "Training error", "Text")[1],
                missing  = "Exclude cases with missing data",
                seed = 12321,
                show.labels = FALSE,
                ...)
{
    if (is.character(hidden))
    {
        h.str <- unlist(strsplit(split=",", hidden))
        hidden <- gsub(" ", "", h.str)
    }
    hidden <- suppressWarnings(as.numeric(hidden))
    if (any(is.na(hidden)))
        stop("'hidden' should be a comma-separated list specifying the number of units in each layer\n")
    if (any(hidden != round(hidden)))
        stop("'hidden' should be a list of integers\n")
    if (length(hidden) < 1)
        stop("hidden must be a vector of at least length 1\n")

    ####################################################################
    ##### Reading in the data and doing some basic tidying        ######
    ####################################################################
    cl <- match.call()
    input.formula <- formula # To work past scoping issues in car package: https://cran.r-project.org/web/packages/car/vignettes/embedding.pdf.
    subset.description <- try(deparse(substitute(subset)), silent = TRUE) #We don't know whether subset is a variable in the environment or in data.
    subset <- eval(substitute(subset), data, parent.frame())
    if (!is.null(subset))
    {
        if (is.null(subset.description) | (class(subset.description) == "try-error") | !is.null(attr(subset, "name")))
            subset.description <- Labels(subset)
        if (is.null(attr(subset, "name")))
            attr(subset, "name") <- subset.description
    }
    if(!is.null(weights))
        if (is.null(attr(weights, "name")))
            attr(weights, "name") <- deparse(substitute(weights))
    weights <- eval(substitute(weights), data, parent.frame())
    data <- GetData(input.formula, data, auxiliary.data = NULL)
    row.names <- rownames(data)
    outcome.name <- OutcomeName(input.formula)
    outcome.i <- match(outcome.name, names(data))
    outcome.variable <- data[, outcome.i]
    numeric.outcome <- !is.factor(outcome.variable)
    variable.labels <- Labels(data)
    outcome.label <- variable.labels[outcome.i]
    if (outcome.label == "data[, outcome.name]")
        outcome.label <- outcome.name
    if (!is.null(weights) & length(weights) != nrow(data))
        stop("'weights' and 'data' are required to have the same number of observations. They do not.")
    if (!is.null(subset) & length(subset) > 1 & length(subset) != nrow(data))
        stop("'subset' and 'data' are required to have the same number of observations. They do not.")

    # Treatment of missing values.
    processed.data <- EstimationData(input.formula, data, subset, weights, missing,seed = seed)
    unfiltered.weights <- processed.data$unfiltered.weights
    .estimation.data <- processed.data$estimation.data
    n.predictors <- ncol(.estimation.data)
    n <- nrow(.estimation.data)
    #if (n < ncol(.estimation.data) + 1)
    #    stop("The sample size is too small for it to be possible to conduct the analysis.")
    post.missing.data.estimation.sample <- processed.data$post.missing.data.estimation.sample
    .weights <- processed.data$weights
    .formula <- DataFormula(input.formula)
    # Resampling to generate a weighted sample, if necessary.
    .estimation.data.1 <- if (is.null(weights)) .estimation.data
                          else AdjustDataToReflectWeights(.estimation.data, .weights)

    ####################################################################
    ##### Fitting the model. Ideally, this should be a call to     #####
    ##### another function, with the output of that function       #####
    ##### called 'original'.                                       #####
    ####################################################################
    cnames <- if (show.labels) Labels(data)
              else colnames(data)
    vnames <- c()
    for (i in 2:ncol(data))
    {
        tmp <- c()
        if (is.ordered(data[,i]))
        {
            tmp <- paste0(":", levels(.estimation.data[,i])[-1])
        } else if (is.factor(data[,i]))
        {
            tmp <- paste0(":", levels(.estimation.data[,i]))
        }
        vnames <- c(vnames, paste0(cnames[i], tmp))
    }

    # Outcome variable is never converted to indicator!
    mmin <- min(as.numeric(.estimation.data.1[,1]), na.rm=T)
    estimation.data.2 <- cbind(as.numeric(.estimation.data.1[,1]),
                               dataToNumeric(.estimation.data.1[,-1]))
    #if (length(vnames) != ncol(estimation.data.2) - 1)
    #       stop("Number of variables differ\n")

    # Class labels MUST start at zero
    if (!numeric.outcome)
        estimation.data.2[,1] <- estimation.data.2[,1] - mmin

    colnames(estimation.data.2)[1] <- outcome.name
    estimation.data.2 <- as.matrix(estimation.data.2)

    # Construct network
    num.output.units <- ifelse(numeric.outcome, 1, length(unique(estimation.data.2[,1])))
    loss.func <- if (numeric.outcome) mx.metric.rmse
                 else mx.metric.accuracy

    net <- mx.symbol.Variable("data")
    for (i in 1:(length(hidden)))
    {
        net <- mx.symbol.FullyConnected(net, num_hidden=hidden[i], name=paste0("fc", i))
        net <- mx.symbol.Activation(net, act_type=unit.function, name=paste0("act", i))
    }
    net <- mx.symbol.FullyConnected(net, num_hidden=num.output.units, name=paste0("fc", length(hidden)+1))
    net <- if (numeric.outcome) mx.symbol.LinearRegressionOutput(net)
           else mx.symbol.SoftmaxOutput(net)

    logger <- mx.metric.logger$new()
    obj <- mx.model.FeedForward.create(net, X=estimation.data.2[,-1], y=estimation.data.2[,1],
                                       optimizer=optimizer,
                                       eval.metric=loss.func,
                                       num.round=epochs,
                                       array.batch.size=batch.size,
                                       array.layout="rowmajor",
                                       epoch.end.callback = mx.callback.log.train.metric(10, logger),
                                       verbose=FALSE)

    result <- list()
    result$original <- obj
    result$log <- logger$train
    result$call <- cl


    ####################################################################
    ##### Saving results, parameters, and tidying up               #####
    ####################################################################
    result$subset <- subset <- row.names %in% rownames(.estimation.data)
    result$weights <- unfiltered.weights
    class(result) <- "DeepLearning"
    result$outcome.name <- outcome.name
    result$sample.description <- processed.data$description
    result$n.observations <- n
    result$estimation.data <- estimation.data.2
    result$numeric.outcome <- numeric.outcome
    result$variablenames <- vnames
    if (!numeric.outcome)
        result$outcome.levels <- levels(outcome.variable)

    if (missing == "Imputation (replace missing values with estimates)")
        data <- processed.data$data
    result$model <- data
    result$formula <- input.formula
    result$output <- output
    result$missing <- missing
    result
}

#' \code{VariableImportance}
#' @description Computes relative importance of input variable to the neural network
#' as the sum of the product of raw input-hidden, hidden-output connection weights
#' as proposed by Olden et al. 2004.
#' @param object Object of class \code{"DeepLearning"}
#' @importFrom NeuralNetTools olden
#' @export
VariableImportance <- function(object)
{
    if (class(object) != "DeepLearning")
        stop("Object should be of class \"DeepLearning\"\n")

    xx <- object$original$arg.params
    mod_in <- c()
    struct <- c()
    i <- 1
    last.i <- NA
    while (i <= length(xx))
    {
        tmp <- as.array(xx[[i]])
        tmp2 <- as.array(xx[[i+1]])
        mod_in <- c(mod_in, as.numeric(rbind(tmp,tmp2)))
        struct <- c(struct, nrow(xx[[i]]))
        last.i <- i
        i <- i + 2  # every second entry is the bias
    }
    struct <- c(struct, ncol(xx[[last.i]]))

    varImp <- suppressWarnings(olden(mod_in, struct=struct, bar_plot=FALSE))
    rownames(varImp) <- colnames(object$estimation.data)[-1]
    #rownames(varImp) <- object$variablenames   # changes depending on show.labels
    return(varImp)
}

#' \code{ConfusionMatrix.DeepLearning}
#' @param obj A model with an categorical outcome variable.
#' @param subset An optional vector specifying a subset of observations to be
#'   used in the fitting process, or, the name of a variable in \code{data}. It
#'   may not be an expression.
#' @param weights An optional vector of sampling weights, or, the name or, the
#'   name of a variable in \code{data}. It may not be an expression.
#' @details The proportion of observed values that take the same values as the predicted values.
#' Where the outcome
#' variable in the model is not a factor and not a count, predicted values are assigned to the closest observed
#' value.
#' @importFrom methods is
#' @importFrom flipRegression ConfusionMatrixFromVariables
#' @export
ConfusionMatrix.DeepLearning <- function(obj, subset = NULL, weights = NULL)
{
    observed <- obj$model[,1]
    predicted <- predict(obj)
    #if(obj$numeric.outcome)
    #    stop("ConfusionMatrix only defined for categorical outcome variables\n")
    return(ConfusionMatrixFromVariables(observed, predicted, subset, weights))
}

#' \code{print.DeepLearning}
#' @importFrom flipFormat DeepLearningTable FormatWithDecimals ExtractCommonPrefix
#' @importFrom plotly plot_ly layout
#' @export
print.DeepLearning <- function(x, ...)
{
    # Check about using label/names
    # Also, perhaps stats should be computed by predicting on the estimation data
    if (x$output == "Training error")
    {
        yy <- x$log
        pp <- plot_ly(x=1:length(yy), y=yy, type="scatter", mode="lines")
        pp <- layout(pp, xaxis=list(title="Training epoch"),
                     yaxis=list(title=ifelse(x$numeric.outcome, "RMSE", "Accuracy")))
        print(pp)

    } else if (x$output == "Summary")
    {
        title <- paste0("Deep Learning: ", x$outcome.name)  # doesn't handle show labels
        imp <- VariableImportance(x)
        extracted <- ExtractCommonPrefix(rownames(imp))
        if (!is.na(extracted$common.prefix))
        {
            title <- paste0(title, " by ", extracted$common.prefix)
            rownames(imp) <- extracted$shortened.labels
        }
        subtitle <- ""
        if (!x$numeric.outcome)
        {
            confM <- ConfusionMatrix(x)
            tot.cor <- sum(diag(confM))/sum(confM)
            class.cor <- unlist(lapply(1:nrow(confM), function(i) {confM[i,i]/sum(confM[i,])}))
            tmp.text <- paste(paste0(rownames(confM), ":"),
                              paste0(FormatWithDecimals(class.cor*100, 2), "%"),
                              collapse=", ")
            subtitle <- sprintf("Correct predictions: %.2f%% (%s)",
                                tot.cor*100, tmp.text)
        } else
        {
            pred <- predict(x)
            rmse <- sqrt(mean((pred - x$model[,1])^2))
            rsq <- (cor(pred, x$model[,1]))^2
            subtitle <- sprintf("R-sq: %.2f, RMSE: %.2f", rsq, rmse)
        }
        tbl <- DeepLearningTable(imp,
                                 order.values = TRUE,
                                 title = title,
                                 subtitle = subtitle,
                                 footer = x$sample.description)
        print(tbl)
    } else
    {
        print(x$call)
        cat(x$sample.description, "\n")

        if (!x$numeric.outcome)
        {
            cat("\nConfusion matrix:\n")
            print(ConfusionMatrix.DeepLearning(x))
            cat("\n")
        }

        vImp <- VariableImportance(x)
        cat("\nVariable importance (Olden)\n")
        print(vImp[order(vImp[,1], decreasing=T),,drop=FALSE])
    }
}


