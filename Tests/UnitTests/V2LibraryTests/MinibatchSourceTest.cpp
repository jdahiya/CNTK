//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKLibrary.h"
#include "Common.h"

using namespace CNTK;

// Mock communicator to simulate MPI run
class MockCommunicator : public DistributedCommunicator
{
private:
    std::unordered_set<DistributedWorkerDescriptor> m_workers;
    DistributedWorkerDescriptor m_self;

public:
    virtual const std::unordered_set<DistributedWorkerDescriptor>& Workers() const override
    {
        return m_workers;
    }

    virtual const DistributedWorkerDescriptor& CurrentWorker() const override
    {
        return m_self;
    }

    virtual DistributedCommunicatorPtr SubGroup(const std::unordered_set<DistributedWorkerDescriptor>&) const override
    {
        return nullptr;
    }

    virtual void Concatenate(
        const std::vector<ValuePtr>&,
        std::vector<ValuePtr>&,
        const std::unordered_set<DistributedWorkerDescriptor>&) override
    {}

    virtual void Concatenate(
        const std::vector<NDArrayViewPtr>&,
        std::vector<NDArrayViewPtr>&,
        const std::unordered_set<DistributedWorkerDescriptor>&) override
    {}

    virtual void Gather(
        const Dictionary&,
        std::vector<DictionaryPtr>&,
        const std::unordered_set<DistributedWorkerDescriptor>&) override
    {}

    virtual void AggregateInPlace(
        const std::vector<NDArrayViewPtr>&,
        const std::unordered_set<DistributedWorkerDescriptor>&) override
    {}

    virtual void Aggregate(
        const std::vector<NDArrayViewPtr>&,
        std::vector<NDArrayViewPtr>&,
        const std::unordered_set<DistributedWorkerDescriptor>&) override
    {}
    
    virtual void Barrier() override
    {}

    MockCommunicator(size_t numWorkers)
    {
        for (size_t i = 0; i < numWorkers; i++)
        {
            DistributedWorkerDescriptor desc;
            desc.m_hostId = L"MockCommunicator";
            desc.m_globalRank = i;

            m_workers.insert(desc);
        }
        MockRank(0);
    }

    void MockRank(size_t rank)
    {
        m_self.m_hostId = L"MockCommunicator";
        m_self.m_globalRank = rank;
    }
};

MinibatchSourcePtr TextFormatMinibatchSource(const std::wstring& dataFilePath, const std::vector<StreamConfiguration>& streamConfigs, size_t epochSize, bool randomize, size_t chunkSizeInBytes)
{
    ::CNTK::Dictionary minibatchSourceConfiguration;
    minibatchSourceConfiguration[L"epochSize"] = epochSize;

    if (randomize)
        minibatchSourceConfiguration[L"randomize"] = true;
    else
        minibatchSourceConfiguration[L"randomize"] = false;

    ::CNTK::Dictionary deserializerConfiguration;
    deserializerConfiguration[L"type"] = L"CNTKTextFormatDeserializer";
    deserializerConfiguration[L"file"] = dataFilePath;

    ::CNTK::Dictionary inputStreamsConfig;
    for (auto streamConfig : streamConfigs)
    {
        std::wstring streamName = streamConfig.m_streamName;
        size_t streamDim = streamConfig.m_dim;
        bool isSparse = streamConfig.m_isSparse;
        std::wstring streamAlias = streamConfig.m_streamAlias;

        ::CNTK::Dictionary inputStreamConfig;
        inputStreamConfig[L"dim"] = streamDim;
        inputStreamConfig[L"format"] = isSparse ? L"sparse" : L"dense";
        if (!streamAlias.empty())
            inputStreamConfig[L"alias"] = streamAlias;

        inputStreamsConfig[streamName] = inputStreamConfig;
    }

    deserializerConfiguration[L"input"] = inputStreamsConfig;
    deserializerConfiguration[L"chunkSizeInBytes"] = chunkSizeInBytes;
    minibatchSourceConfiguration[L"deserializers"] = std::vector<::CNTK::DictionaryValue>({ deserializerConfiguration });
    return CreateCompositeMinibatchSource(minibatchSourceConfiguration);
}

void TestMinibatchSourceWarmStart(size_t minibatchSize, size_t warmStartSamples, bool randomize, size_t chunkSizeInBytes, bool expectNoData = false)
{
    // TODO: Currently this test is based on the number of samples.
    // We should switch to the real data instead.

    const size_t inputDim = 2;
    const size_t numOutputClasses = 2;
    auto featureStreamName = L"features";
    auto labelsStreamName = L"labels";

    auto groundTruth = TextFormatMinibatchSource(
        L"SimpleDataTrain_cntk_text.txt",
        { { featureStreamName, inputDim }, { labelsStreamName, numOutputClasses } },
        MinibatchSource::FullDataSweep,
        randomize,
        chunkSizeInBytes);

    auto featureStreamInfo = groundTruth->StreamInfo(featureStreamName);
    auto labelStreamInfo = groundTruth->StreamInfo(labelsStreamName);

    // Read all data as it should be without any workers.
    bool hasData = true;
    size_t numberOfSamplesInSweep = 0;
    while (hasData)
    {
        auto minibatchData = groundTruth->GetNextMinibatch(minibatchSize);
        if (minibatchData.empty())
        {
            hasData = false;
            continue;
        }
        numberOfSamplesInSweep += minibatchData[featureStreamInfo].m_numSamples;
    }

    // Let's create two workers.
    auto minibatchSource = TextFormatMinibatchSource(
        L"SimpleDataTrain_cntk_text.txt",
        { { featureStreamName, inputDim }, { labelsStreamName, numOutputClasses } },
        MinibatchSource::FullDataSweep,
        randomize,
        chunkSizeInBytes);

    auto minibatchSource2 = TextFormatMinibatchSource(
        L"SimpleDataTrain_cntk_text.txt",
        { { featureStreamName, inputDim }, { labelsStreamName, numOutputClasses } },
        MinibatchSource::FullDataSweep,
        randomize,
        chunkSizeInBytes);

    size_t totalSamples = 0;
    hasData = true;
    while (hasData)
    {
        if (totalSamples < warmStartSamples)
        {
            auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, 0, 1, 0);
            auto minibatchData2 = minibatchSource2->GetNextMinibatch(minibatchSize, 0, 1, 0);

            if (minibatchData[featureStreamInfo].m_numSamples != minibatchData2[featureStreamInfo].m_numSamples)
                ReportFailure("Data does not match, reads are not deterministic!!!");

            // Because they are supposed to read the same data - adding it only once.
            totalSamples += minibatchData[featureStreamInfo].m_numSamples;
        }
        else
        {
            // We are in distributed mode, the sum should be equal to the minibatch size
            // or less at the end of the sweep.
            auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, 0, 2, 0);
            auto minibatchData2 = minibatchSource2->GetNextMinibatch(minibatchSize, 0, 2, 1);

            hasData = !minibatchData.empty() || !minibatchData2.empty();
            if (!hasData)
                break;

            // Update the counter
            size_t accumulative = 0;
            if (!minibatchData.empty())
                accumulative += minibatchData[featureStreamInfo].m_numSamples;
            if (!minibatchData2.empty())
                accumulative += minibatchData2[featureStreamInfo].m_numSamples;

            totalSamples += accumulative;

            if (expectNoData) // second worker does not have any data.
            {
                if (minibatchData[featureStreamInfo].m_numSamples != minibatchSize/2 && totalSamples != numberOfSamplesInSweep)
                    ReportFailure("TestMinibatchSourceWarmStart failed because data did not match."
                                  "Expected minibatch size '%d', acutal '%d'. Total number of sample '%d', sweep '%d'.",
                                  minibatchSize,
                                  minibatchData[featureStreamInfo].m_numSamples,
                                  totalSamples,
                                  numberOfSamplesInSweep);
            }
            else
            {
                if (accumulative != minibatchSize && totalSamples != numberOfSamplesInSweep)
                    ReportFailure("TestMinibatchSourceWarmStart failed because data did not match."
                        "Expected minibatch size '%d', acutal '%d'. Total number of sample '%d', sweep '%d'.",
                        minibatchSize,
                        accumulative,
                        totalSamples,
                        numberOfSamplesInSweep);
            }
        }
    }

    if (totalSamples != numberOfSamplesInSweep)
        ReportFailure("Expected sweep number '%d' did not match the actual '%d'.",
                      numberOfSamplesInSweep,
                      totalSamples);
}

void MinibatchSourceTests()
{
    // Test no-randomize minibatch source with small data chunks
    TestMinibatchSourceWarmStart(64, 128, false, 1024);
    TestMinibatchSourceWarmStart(64, 0, false, 1024);
    TestMinibatchSourceWarmStart(64, 100, false, 1024);

    // Test no-randomized minibatch source with a single chunk
    size_t chunk32MB = 1024 * 1024 * 32;
    TestMinibatchSourceWarmStart(64, 128, false, chunk32MB);
    TestMinibatchSourceWarmStart(64, 0, false, chunk32MB);
    TestMinibatchSourceWarmStart(64, 100, false, chunk32MB);

    // Test randomized minibatch source with small data chunks
    TestMinibatchSourceWarmStart(64, 0, true, 1024);
    TestMinibatchSourceWarmStart(64, 128, true, 1024);

    // Test randomized minibatch source with no data for one of the workers 
    // due to decimation based on chunks
    bool expectNoData = true;
    TestMinibatchSourceWarmStart(64, 0, true, chunk32MB, expectNoData);
    TestMinibatchSourceWarmStart(64, 128, true, chunk32MB, expectNoData);
}