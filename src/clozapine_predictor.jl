# data columns:
# - male: 0/1
# - age: Float
# - dose: Float
# - bmi: Float
# - weight: Float
# - duration of clozapine treatment [days]: Float
# - crp: Float
# - inducers_3a4: Int
# - inhibitors_3a4: Int
# - substrates_3a4: Int
# - inducers_1a2: Int
# - inhibitors_1a2: Int
# - substrates_1a2: Int

println("Importing packages")
using CSV
using DataFrames
using MLJ
using MLJDecisionTreeInterface
using MLJLinearModels
using Random
using StatsBase
using JLD2

# load training data
if isfile("clozapine_test.csv")
    println("Loading raw data: clozapine_test.csv")
    predict_raw_data = CSV.read("clozapine_test.csv", header=true, DataFrame)
else
    error("File clozapine_test.csv cannot be opened!")
    exit(-1)
end

# load models
if isfile("clozapine_classifier_model.jlso")
    println("Loading model: clozapine_classifier_model.jlso")
    model_rfc = machine("clozapine_classifier_model.jlso")
else
    error("File clozapine_classifier_model.jlso cannot be opened!")
    exit(-1)
end
if isfile("clozapine_regressor_model.jlso")
    println("Loading model: clozapine_regressor_model.jlso")
    clo_model_rfr = machine("clozapine_regressor_model.jlso")
else
    error("File clozapine_regressor_model.jlso cannot be opened!")
    exit(-1)
end
if isfile("norclozapine_regressor_model.jlso")
    println("Loading model: norclozapine_regressor_model.jlso")
    nclo_model_rfr = machine("norclozapine_regressor_model.jlso")
else
    error("File norclozapine_regressor_model.jlso cannot be opened!")
    exit(-1)
end
if isfile("scaler.jld")
    println("Loading: scaler.jld")
    scaler = JLD2.load_object("scaler.jld")
else
    error("File scaler.jld cannot be opened!")
    exit(-1)
end

function preprocess(predict_raw_data)
    y1 = predict_raw_data[:, 1]
    y2 = string.(predict_raw_data[:, 2])
    y3 = predict_raw_data[:, 3]
    replace!(y2, "0" => "norm")
    replace!(y2, "1" => "high")
    x = Matrix(predict_raw_data[:, 5:end])
    return x, y1, y2, y3
end

function print_confusion_matrix(cm)
    println("""
                     group
                   norm   high
                 ┌──────┬──────┐
            norm │ $(lpad(cm[1], 4, " ")) │ $(lpad(cm[3], 4, " ")) │
 prediction      ├──────┼──────┤
            high │ $(lpad(cm[2], 4, " ")) │ $(lpad(cm[4], 4, " ")) │
                 └──────┴──────┘
             """)
end

x, y1, y2, y3 = preprocess(predict_raw_data)

# standardize
println("Processing: standardize")
# data = hcat(y, x[:, 2:end])
data = x[:, 2:7]
# scaler = StatsBase.fit(ZScoreTransform, data, dims=1)
# data = StatsBase.transform(scaler, data)
# or
m = scaler.mean
s = scaler.scale
for idx in 1:size(data, 1)
    data[idx, :] = (data[idx, :] .- m) ./ s
end
data[isnan.(data)] .= 0
x_gender = Bool.(x[:, 1])
x_cont = data
x_rest = x[:, 8:end]
x1 = DataFrame(:male=>x_gender)
x2 = DataFrame(x_cont, ["age", "dose", "bmi", "weight", "duration", "crp"])
x2 = DataFrame(x_cont[:, [1, 2, 3, 6]], ["age", "dose", "bmi", "crp"])
x3 = DataFrame(x_rest, ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
x = hcat(x1, x2, x3)
# x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :weight=>Continuous, :duration=>Continuous, :crp=>Continuous, :inducers_3a4=>Count,  :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)
x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Count, :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)
y2 = DataFrame(group=y2)
y2 = coerce(y2.group, OrderedFactor{2})
# scitype(y)
# levels(y)
println("Number of entries: $(size(y1, 1))")
println("")
println("Calculating predictions")
println("-----------------------")
println("")
println("Regressor:")
yhat1 = MLJ.predict(clo_model_rfr, x)
yhat3 = MLJ.predict(nclo_model_rfr, x)
yhat1 = round.(yhat1, digits=1)
yhat3 = round.(yhat3, digits=1)
rmse_clo = zeros(length(yhat1))
rmse_nclo = zeros(length(yhat3))
for idx in eachindex(yhat1)
    rmse_clo[idx] = round.(sqrt((y1[idx] - yhat1[idx])^2), digits=2)
    rmse_nclo[idx] = round.(sqrt((y3[idx] - yhat3[idx])^2), digits=2)
    println("Subject ID: $idx \t CLO level: $(y1[idx]) \t prediction: $(yhat1[idx]) \t RMSE: $(rmse_clo[idx])")
    println("Subject ID: $idx \t NCLO level: $(y3[idx]) \t prediction: $(yhat3[idx]) \t RMSE: $(rmse_nclo[idx])")
    println()
end
println()
yhat2 = MLJ.predict(model_rfc, x)
subj1 = 0
subj2 = 0
subj3 = 0
subj4 = 0
println("Classifier:")
for idx in eachindex(yhat2)
    print("Subject ID: $idx \t level: $(uppercase(String(y2[idx]))) \t")
    p_high = broadcast(pdf, yhat2[idx], "high")
    p_norm = broadcast(pdf, yhat2[idx], "norm")
    if p_norm > p_high
        print("prediction: NORM, prob = $(round(p_norm, digits=2)) \t")
        if String(y2[idx]) == "norm"
            global subj1 += 1
        elseif String(y2[idx]) == "high"
            global subj2 += 1
        end
    else
        print("prediction: HIGH, prob = $(round(p_high, digits=2)) \t")
        if String(y2[idx]) == "high"
            global subj4 += 1
        elseif String(y2[idx]) == "norm"
            global subj3 += 1
        end
    end
    if yhat1[idx] > 550
        p_high += 0.5
        p_norm -= 0.5
    elseif yhat1[idx] <= 550
        p_norm += 0.5
        p_high -= 0.5
    end
    if yhat3[idx] > 400
        p_high += 0.5
        p_norm -= 0.5
    elseif yhat3[idx] <= 400
        p_norm += 0.5
        p_high -= 0.5
    end
    p_high > 1.0 && (p_high = 1.0)
    p_high < 0.0 && (p_high = 0.0)
    p_norm > 1.0 && (p_norm = 1.0)
    p_norm < 0.0 && (p_norm = 0.0)
    if p_norm > p_high
        println("adj. prediction: NORM, prob = $(round(p_norm, digits=2))")
        if String(y2[idx]) == "norm"
            global subj1 += 1
        elseif String(y2[idx]) == "high"
            global subj2 += 1
        end
    else
        println("adj. prediction: HIGH, prob = $(round(p_high, digits=2))")
        if String(y2[idx]) == "high"
            global subj4 += 1
        elseif String(y2[idx]) == "norm"
            global subj3 += 1
        end
    end
end

println()
println("Prediction accuracy:")
m = RSquared()
println("R²: ", round(m(yhat1, y1), digits=4))
m = RootMeanSquaredError()
println("RMSE: ", round(m(yhat1, y1), digits=4))
mcr = round(100 * ((subj2 + subj3) / 10), digits=2)
println("Miss-classification rate: $mcr %")
println("Regressor execution time and memory use:")
@time yhat = MLJ.predict(clo_model_rfr, x)
println("Classifier execution time and memory use:")
@time yhat2 = MLJ.predict(model_rfc, x)
println()
print_confusion_matrix([subj1 subj2; subj3 subj4])
println("Analysis completed.")

function predict_single_patient(patient_data::Vector{Float64})

    m = scaler.mean
    s = scaler.scale
    patient_data[2:7] = (patient_data[2:7] .- m) ./ s
    patient_data[isnan.(patient_data)] .= 0
    x_gender = Bool(patient_data[1])
    x_cont = patient_data[2:7]
    x_rest = patient_data[8:end]
    x1 = DataFrame(:male=>x_gender)
    # x2 = DataFrame(x_cont, ["age", "dose", "bmi", "weight", "duration", "crp"])
    x_cont = x_cont[[1, 2, 3, 6]]
    x2 = DataFrame(reshape(x_cont, 1, length(x_cont)), ["age", "dose", "bmi", "crp"])
    x3 = DataFrame(reshape(x_rest, 1, length(x_rest)), ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
    x = hcat(x1, x2, x3)
    # x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :weight=>Continuous, :duration=>Continuous, :crp=>Continuous, :inducers_3a4=>Count,  :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)
    x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Count, :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)

    yhat1 = MLJ.predict(clo_model_rfr, x)[1]
    yhat1 = round(yhat1, digits=1)
    yhat3 = MLJ.predict(nclo_model_rfr, x)[1]
    yhat3 = round(yhat3, digits=1)
    println("Predicted CLO level: $(yhat1)")
    println("Predicted NCLO level: $(yhat3)")
    yhat2 = MLJ.predict(model_rfc, x)[1]
    print("Predicted group: ")
    p_high = broadcast(pdf, yhat2, "high")
    p_norm = broadcast(pdf, yhat2, "norm")
    if p_norm > p_high
        println("NORM, prob = $(round(p_norm, digits=2))")
    else
        println("HIGH, prob = $(round(p_high, digits=2))")
    end
    if yhat1 > 550
        p_high += 0.5
        p_norm -= 0.5
    elseif yhat1 <= 550
        p_norm += 0.5
        p_high -= 0.5
    end
    if yhat3 > 400
        p_high += 0.5
        p_norm -= 0.5
    elseif yhat3 <= 400
        p_norm += 0.5
        p_high -= 0.5
    end
    p_high > 1.0 && (p_high = 1.0)
    p_high < 0.0 && (p_high = 0.0)
    p_norm > 1.0 && (p_norm = 1.0)
    p_norm < 0.0 && (p_norm = 0.0)
    if p_norm > p_high
        println("Adjusted prediction: NORM, prob = $(round(p_norm, digits=2))")
    else
        println("Adjusted prediction: HIGH, prob = $(round(p_high, digits=2))")
    end
end

function recommended_dose(patient_data::Vector{Float64})
    # - male: 0/1
    # - age: Float
    # + dose: Float
    # - bmi: Float
    # + weight: Float
    # + duration of clozapine treatment [days]: Float
    # - crp: Float
    # - inducers_3a4: Int
    # - inhibitors_3a4: Int
    # - substrates_3a4: Int
    # - inducers_1a2: Int
    # - inhibitors_1a2: Int
    # - substrates_1a2: Int

    doses = 12.5:12.5:1000

    for idx in eachindex(doses)
        data = patient_data
        data = vcat(data[1:2], doses[idx], data[3:end])
        data[2:7] = (data[2:7] .- m) ./ s
        data[isnan.(data)] .= 0
        x_gender = Bool(data[1])
        x_cont = data[2:7]
        x_rest = data[8:end]
        x1 = DataFrame(:male=>x_gender)
        # x2 = DataFrame(x_cont, ["age", "dose", "bmi", "weight", "duration", "crp"])
        x_cont = x_cont[[1, 2, 3, 6]]
        x2 = DataFrame(reshape(x_cont, 1, length(x_cont)), ["age", "dose", "bmi", "crp"])
        x3 = DataFrame(reshape(x_rest, 1, length(x_rest)), ["inducers_3a4", "inhibitors_3a4", "substrates_3a4", "inducers_1a2", "inhibitors_1a2", "substrates_1a2"])
        x = hcat(x1, x2, x3)
        # x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :weight=>Continuous, :duration=>Continuous, :crp=>Continuous, :inducers_3a4=>Count,  :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)
        x = coerce(x, :age=>Multiclass, :dose=>Continuous, :bmi=>Continuous, :crp=>Continuous, :inducers_3a4=>Count, :inhibitors_3a4=>Count, :substrates_3a4=>Count, :inducers_1a2=>Count, :inhibitors_1a2=>Count, :substrates_1a2=>Count)

        yhat1 = MLJ.predict(clo_model_rfr, x)[1]
        yhat1 = round(yhat1, digits=1)
        yhat3 = MLJ.predict(nclo_model_rfr, x)[1]
        yhat3 = round(yhat3, digits=1)
        println("CLO dose: $(doses[idx]) mg/d")
        println("\t Predicted CLO level: $(yhat1)")
        println("\t Predicted NCLO level: $(yhat3)")
        yhat2 = MLJ.predict(model_rfc, x)[1]
        print("\t Predicted group: ")
        p_high = broadcast(pdf, yhat2, "high")
        p_norm = broadcast(pdf, yhat2, "norm")
        if p_norm > p_high
            println("NORM, prob = $(round(p_norm, digits=2))")
        else
            println("HIGH, prob = $(round(p_high, digits=2))")
        end
        if yhat1 > 550
            p_high += 0.5
            p_norm -= 0.5
        elseif yhat1 <= 550
            p_norm += 0.5
            p_high -= 0.5
        end
        if yhat3 > 400
            p_high += 0.5
            p_norm -= 0.5
        elseif yhat3 <= 400
            p_norm += 0.5
            p_high -= 0.5
        end
        p_high > 1.0 && (p_high = 1.0)
        p_high < 0.0 && (p_high = 0.0)
        p_norm > 1.0 && (p_norm = 1.0)
        p_norm < 0.0 && (p_norm = 0.0)
        if p_norm > p_high
            println("\t Adjusted prediction: NORM, prob = $(round(p_norm, digits=2))")
        else
            println("\t Adjusted prediction: HIGH, prob = $(round(p_high, digits=2))")
        end
    end
end

predict_single_patient([0,58,150,23.18,67,93,3,0,0,0,0,0,1])
predict_single_patient([0,60,400,25,84.5,16,2.3,0,2,2,0,0,1])
recommended_dose([1,90,25,84.5,16,2.3,0,2,2,0,0,1])
